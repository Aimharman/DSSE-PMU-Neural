/*
 * loopback_combined.c
 *
 * Single-process transmitter + receiver: generates Im*sin(wt) as SPWM on
 * one GPIO and simultaneously captures/reconstructs it from another GPIO,
 * writing the result to CSV.
 *
 * Running transmitter and receiver as two separate processes does NOT work
 * reliably: pigpio only supports one process driving the GPIO/DMA hardware
 * at a time, so a second process contending for it produces corrupted or
 * all-zero readings. This program avoids that by doing both jobs in one
 * process (one pigpio instance, two threads).
 *
 * Duty-cycle capture is done by a dedicated busy-poll thread reading the
 * input pin's raw level (gpioRead, a direct register read) as fast as
 * possible, rather than via pigpio's DMA/alert edge notifications: those
 * share the same clock hardware as gpioHardwarePWM and were found to stop
 * reporting edges once hardware PWM was active, corrupting duty readings.
 *
 * Requires: pigpio (http://abyz.me.uk/rpi/pigpio/), run as root.
 *
 * Build:  make
 * Run:    sudo ./loopback_combined -o 18 -i 23 -f 50 -c 20000 -a 1.0 -r 1000 -d 5 -O capture.csv
 *
 * Options:
 *   -o <pin>    Output GPIO driving the sine (hardware PWM capable)   default 18
 *   -i <pin>    Input GPIO wired via jumper to the output pin         default 23
 *   -f <Hz>     Sine frequency                                        default 50
 *   -c <Hz>     PWM carrier frequency                                 default 20000
 *   -a <Im>     Peak amplitude, normalized 0.0-1.0                    default 1.0
 *   -r <Hz>     Sample/bin rate (20 samples/cycle at 50 Hz by default)  default 1000
 *   -d <sec>    Duration in seconds                                   default 5
 *   -O <file>   Output CSV path                                       default capture.csv
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <signal.h>
#include <unistd.h>
#include <pthread.h>
#include <stdatomic.h>
#include <pigpio.h>

#define PWM_RANGE 1000000

static volatile sig_atomic_t g_running = 1;
static volatile int g_sampler_stop = 0;
static int g_sampler_gpio = 23;

static atomic_ullong g_high_count = 0;
static atomic_ullong g_total_count = 0;

static void handle_sigint(int signum) {
    (void)signum;
    g_running = 0;
}

/* busy-poll the input pin as fast as possible: plain register reads, no
 * dependency on pigpio's DMA/alert notification path. */
static void *sampler_thread_fn(void *arg) {
    (void)arg;
    while (!g_sampler_stop) {
        int lvl = gpioRead((unsigned)g_sampler_gpio);
        atomic_fetch_add_explicit(&g_total_count, 1, memory_order_relaxed);
        if (lvl) atomic_fetch_add_explicit(&g_high_count, 1, memory_order_relaxed);
    }
    return NULL;
}

int main(int argc, char **argv) {
    int out_gpio = 18;
    int in_gpio = 23;
    double sine_freq = 50.0;
    unsigned carrier_freq = 20000;
    double amplitude = 1.0;
    double sample_rate = 1000.0;
    double duration = 5.0;
    const char *out_path = "capture.csv";

    int opt;
    while ((opt = getopt(argc, argv, "o:i:f:c:a:r:d:O:h")) != -1) {
        switch (opt) {
            case 'o': out_gpio = atoi(optarg); break;
            case 'i': in_gpio = atoi(optarg); break;
            case 'f': sine_freq = atof(optarg); break;
            case 'c': carrier_freq = (unsigned)atoi(optarg); break;
            case 'a': amplitude = atof(optarg); break;
            case 'r': sample_rate = atof(optarg); break;
            case 'd': duration = atof(optarg); break;
            case 'O': out_path = optarg; break;
            default:
                fprintf(stderr,
                    "Usage: %s [-o out_gpio] [-i in_gpio] [-f sine_hz] [-c carrier_hz] "
                    "[-a amplitude] [-r sample_hz] [-d duration_s] [-O out.csv]\n", argv[0]);
                return 1;
        }
    }

    if (out_gpio == in_gpio) {
        fprintf(stderr, "Output and input gpio must differ\n");
        return 1;
    }
    if (amplitude <= 0.0 || amplitude > 1.0) {
        fprintf(stderr, "Amplitude must be in (0.0, 1.0]\n");
        return 1;
    }
    if (sample_rate <= 2.0 * sine_freq) {
        fprintf(stderr, "Sample rate must exceed twice the sine frequency (Nyquist)\n");
        return 1;
    }

    FILE *fp = fopen(out_path, "w");
    if (!fp) {
        perror("fopen");
        return 1;
    }
    fprintf(fp, "Time (s),TX Value,Duty Cycle,Reconstructed Value,Poll Samples\n");

    if (gpioInitialise() < 0) {
        fprintf(stderr, "pigpio initialisation failed (run as root, and make sure no other "
                         "pigpio-using process is running)\n");
        fclose(fp);
        return 1;
    }
    signal(SIGINT, handle_sigint);

    gpioSetMode((unsigned)in_gpio, PI_INPUT);
    gpioSetPullUpDown((unsigned)in_gpio, PI_PUD_DOWN);

    g_sampler_gpio = in_gpio;
    pthread_t sampler_thread;
    if (pthread_create(&sampler_thread, NULL, sampler_thread_fn, NULL) != 0) {
        fprintf(stderr, "failed to start sampler thread\n");
        gpioTerminate();
        fclose(fp);
        return 1;
    }

    const double dt = 1.0 / sample_rate;
    const double w = 2.0 * M_PI * sine_freq;
    double t = 0.0;

    fprintf(stderr,
        "Looping back: out=gpio%d in=gpio%d f=%.2fHz carrier=%uHz Im=%.2f rate=%.1fHz -> %s\n",
        out_gpio, in_gpio, sine_freq, carrier_freq, amplitude, sample_rate, out_path);

    while (g_running && t < duration) {
        double tx_value = amplitude * sin(w * t);
        double tx_duty_frac = (tx_value + amplitude) / (2.0 * amplitude);
        unsigned duty = (unsigned)(tx_duty_frac * PWM_RANGE);

        if (gpioHardwarePWM((unsigned)out_gpio, carrier_freq, duty) != 0) {
            fprintf(stderr, "gpioHardwarePWM failed (check gpio/carrier_freq)\n");
            break;
        }

        time_sleep(dt);

        unsigned long long high = atomic_exchange_explicit(&g_high_count, 0, memory_order_relaxed);
        unsigned long long total = atomic_exchange_explicit(&g_total_count, 0, memory_order_relaxed);
        double rx_duty_frac = (total > 0) ? ((double)high / (double)total) : 0.0;

        double rx_value = amplitude * (2.0 * rx_duty_frac - 1.0);
        fprintf(fp, "%.6f,%.6f,%.6f,%.6f,%llu\n", t, tx_value, rx_duty_frac, rx_value, total);

        t += dt;
    }

    g_sampler_stop = 1;
    pthread_join(sampler_thread, NULL);

    gpioHardwarePWM((unsigned)out_gpio, 0, 0);
    gpioTerminate();
    fclose(fp);
    return 0;
}
