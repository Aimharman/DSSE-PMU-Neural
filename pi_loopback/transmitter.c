/*
 * transmitter.c
 *
 * Generates a sinusoid  i(t) = Im * sin(w*t)  on a Raspberry Pi 4B GPIO pin
 * using hardware PWM: the duty cycle is updated every sample to trace the
 * sine wave (SPWM). The digital PWM output can be read back on another GPIO
 * (square-wave/edge capture) or, if fed through an external RC low-pass
 * filter, reconstructed as an analog sine for scope/ADC measurement.
 *
 * Requires: pigpio (http://abyz.me.uk/rpi/pigpio/) running as root
 *           (sudo pigpiod not required; this uses the pigpio C library directly).
 *
 * Build:  make
 * Run:    sudo ./transmitter -g 18 -f 50 -c 20000 -a 1.0 -r 1000 -d 5
 *
 * Options:
 *   -g <pin>    BCM GPIO number for hardware PWM (12, 13, 18 or 19 on Pi 4B)
 *   -f <Hz>     Sine frequency  (w = 2*pi*f)               default 50
 *   -c <Hz>     PWM carrier frequency                       default 20000
 *   -a <Im>     Peak amplitude, normalized 0.0-1.0           default 1.0
 *   -r <Hz>     Sample rate (20 samples/cycle at 50 Hz)      default 1000
 *   -d <sec>    Duration in seconds, 0 = run until Ctrl+C    default 0
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <signal.h>
#include <unistd.h>
#include <pigpio.h>

#define PWM_RANGE 1000000 /* pigpio hardware PWM duty cycle range: 0-1000000 */

static volatile sig_atomic_t g_running = 1;

static void handle_sigint(int signum) {
    (void)signum;
    g_running = 0;
}

int main(int argc, char **argv) {
    int gpio = 18;
    double sine_freq = 50.0;
    unsigned carrier_freq = 20000;
    double amplitude = 1.0;
    double sample_rate = 1000.0;
    double duration = 0.0;

    int opt;
    while ((opt = getopt(argc, argv, "g:f:c:a:r:d:h")) != -1) {
        switch (opt) {
            case 'g': gpio = atoi(optarg); break;
            case 'f': sine_freq = atof(optarg); break;
            case 'c': carrier_freq = (unsigned)atoi(optarg); break;
            case 'a': amplitude = atof(optarg); break;
            case 'r': sample_rate = atof(optarg); break;
            case 'd': duration = atof(optarg); break;
            default:
                fprintf(stderr,
                    "Usage: %s [-g gpio] [-f sine_hz] [-c carrier_hz] "
                    "[-a amplitude] [-r sample_hz] [-d duration_s]\n", argv[0]);
                return 1;
        }
    }

    if (amplitude <= 0.0 || amplitude > 1.0) {
        fprintf(stderr, "Amplitude must be in (0.0, 1.0]\n");
        return 1;
    }
    if (sample_rate <= 2.0 * sine_freq) {
        fprintf(stderr, "Sample rate must exceed twice the sine frequency (Nyquist)\n");
        return 1;
    }

    if (gpioInitialise() < 0) {
        fprintf(stderr, "pigpio initialisation failed (run as root)\n");
        return 1;
    }
    signal(SIGINT, handle_sigint);

    const double dt = 1.0 / sample_rate;
    const double w = 2.0 * M_PI * sine_freq;
    double t = 0.0;

    fprintf(stderr,
        "Transmitting Im*sin(wt): gpio=%d f=%.2fHz carrier=%uHz Im=%.2f rate=%.1fHz\n",
        gpio, sine_freq, carrier_freq, amplitude, sample_rate);

    while (g_running && (duration <= 0.0 || t < duration)) {
        double value = amplitude * sin(w * t);           /* range [-Im, Im] */
        double duty_frac = (value + amplitude) / (2.0 * amplitude); /* -> [0,1] */
        unsigned duty = (unsigned)(duty_frac * PWM_RANGE);

        if (gpioHardwarePWM((unsigned)gpio, carrier_freq, duty) != 0) {
            fprintf(stderr, "gpioHardwarePWM failed (check gpio/carrier_freq)\n");
            break;
        }

        time_sleep(dt);
        t += dt;
    }

    gpioHardwarePWM((unsigned)gpio, 0, 0); /* stop PWM output */
    gpioTerminate();
    return 0;
}
