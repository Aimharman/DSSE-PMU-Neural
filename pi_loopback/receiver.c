/*
 * receiver.c
 *
 * Reads a Raspberry Pi 4B GPIO input pin wired via jumper to the output of
 * transmitter.c, reconstructs the sine amplitude by averaging the duty
 * cycle of the incoming SPWM signal over fixed time bins, and dumps the
 * result to CSV as  Time (s), Duty Cycle, Reconstructed Value.
 *
 * This mirrors what an RC low-pass filter would do in hardware, but in
 * software: within each bin of length 1/sample_rate, the fraction of time
 * the input was HIGH approximates the transmitter's duty_frac for that
 * bin, from which the original value = Im*(2*duty_frac - 1) is recovered.
 *
 * Requires: pigpio (http://abyz.me.uk/rpi/pigpio/), run as root.
 *
 * Build:  make
 * Run:    sudo ./receiver -g 23 -a 1.0 -r 1000 -d 5 -o capture.csv
 *
 * Options:
 *   -g <pin>    BCM GPIO number wired to the transmitter's output    default 23
 *   -a <Im>     Peak amplitude used by the transmitter (0.0-1.0]     default 1.0
 *   -r <Hz>     Bin rate == transmitter's sample rate                default 1000
 *   -d <sec>    Duration in seconds, 0 = run until Ctrl+C            default 0
 *   -o <file>   Output CSV path                                      default capture.csv
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <pthread.h>
#include <pigpio.h>

static volatile sig_atomic_t g_running = 1;

static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static uint32_t g_last_tick = 0;
static int g_last_level = 0;
static uint64_t g_high_ticks = 0; /* accumulated HIGH time (us) in current bin */
static int g_have_edge = 0;

static void handle_sigint(int signum) {
    (void)signum;
    g_running = 0;
}

/* pigpio alert callback: runs in its own thread on every edge of gpio */
static void edge_cb(int gpio, int level, uint32_t tick) {
    (void)gpio;
    if (level == 2) return; /* watchdog timeout, not a real edge */

    pthread_mutex_lock(&g_lock);
    if (g_have_edge) {
        uint32_t elapsed = tick - g_last_tick; /* handles 32-bit wraparound */
        if (g_last_level == 1) {
            g_high_ticks += elapsed;
        }
    }
    g_last_tick = tick;
    g_last_level = level;
    g_have_edge = 1;
    pthread_mutex_unlock(&g_lock);
}

int main(int argc, char **argv) {
    int gpio = 23;
    double amplitude = 1.0;
    double sample_rate = 1000.0;
    double duration = 0.0;
    const char *out_path = "capture.csv";

    int opt;
    while ((opt = getopt(argc, argv, "g:a:r:d:o:h")) != -1) {
        switch (opt) {
            case 'g': gpio = atoi(optarg); break;
            case 'a': amplitude = atof(optarg); break;
            case 'r': sample_rate = atof(optarg); break;
            case 'd': duration = atof(optarg); break;
            case 'o': out_path = optarg; break;
            default:
                fprintf(stderr,
                    "Usage: %s [-g gpio] [-a amplitude] [-r bin_hz] "
                    "[-d duration_s] [-o out.csv]\n", argv[0]);
                return 1;
        }
    }

    if (amplitude <= 0.0 || amplitude > 1.0) {
        fprintf(stderr, "Amplitude must be in (0.0, 1.0]\n");
        return 1;
    }

    FILE *fp = fopen(out_path, "w");
    if (!fp) {
        perror("fopen");
        return 1;
    }
    fprintf(fp, "Time (s),Duty Cycle,Reconstructed Value\n");

    if (gpioInitialise() < 0) {
        fprintf(stderr, "pigpio initialisation failed (run as root)\n");
        fclose(fp);
        return 1;
    }
    signal(SIGINT, handle_sigint);

    gpioSetMode((unsigned)gpio, PI_INPUT);
    gpioSetAlertFunc(gpio, edge_cb);

    const double bin_dt = 1.0 / sample_rate;
    const uint32_t bin_us = (uint32_t)(bin_dt * 1e6);
    double t = 0.0;

    fprintf(stderr, "Capturing on gpio=%d bin_rate=%.1fHz Im=%.2f -> %s\n",
            gpio, sample_rate, amplitude, out_path);

    while (g_running && (duration <= 0.0 || t < duration)) {
        time_sleep(bin_dt);

        pthread_mutex_lock(&g_lock);
        /* fold in the time from the last edge up to now, then reset the bin */
        if (g_have_edge && g_last_level == 1) {
            uint32_t now = gpioTick();
            g_high_ticks += (now - g_last_tick);
            g_last_tick = now;
        }
        double duty_frac = (double)g_high_ticks / (double)bin_us;
        if (duty_frac > 1.0) duty_frac = 1.0;
        if (duty_frac < 0.0) duty_frac = 0.0;
        g_high_ticks = 0;
        pthread_mutex_unlock(&g_lock);

        double value = amplitude * (2.0 * duty_frac - 1.0);
        fprintf(fp, "%.6f,%.6f,%.6f\n", t, duty_frac, value);

        t += bin_dt;
    }

    gpioSetAlertFunc(gpio, NULL);
    gpioTerminate();
    fclose(fp);
    return 0;
}
