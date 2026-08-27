/*
 * check_loopback.c
 *
 * Verifies that two GPIO pins are physically wired together with a jumper:
 * drives the OUTPUT pin HIGH/LOW several times and checks that the INPUT
 * pin follows each transition. Prints PASS/FAIL per toggle and an overall
 * result.
 *
 * Requires: pigpio (http://abyz.me.uk/rpi/pigpio/), run as root.
 *
 * Build:  make
 * Run:    sudo ./check_loopback -o 18 -i 23
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pigpio.h>

int main(int argc, char **argv) {
    int out_gpio = 18;
    int in_gpio = 23;
    int toggles = 10;
    double settle_s = 0.05; /* time to let the level settle before reading */

    int opt;
    while ((opt = getopt(argc, argv, "o:i:n:h")) != -1) {
        switch (opt) {
            case 'o': out_gpio = atoi(optarg); break;
            case 'i': in_gpio = atoi(optarg); break;
            case 'n': toggles = atoi(optarg); break;
            default:
                fprintf(stderr, "Usage: %s [-o out_gpio] [-i in_gpio] [-n toggles]\n", argv[0]);
                return 1;
        }
    }

    if (out_gpio == in_gpio) {
        fprintf(stderr, "Output and input gpio must differ\n");
        return 1;
    }

    if (gpioInitialise() < 0) {
        fprintf(stderr, "pigpio initialisation failed (run as root)\n");
        return 1;
    }

    gpioSetMode((unsigned)out_gpio, PI_OUTPUT);
    gpioSetMode((unsigned)in_gpio, PI_INPUT);
    gpioSetPullUpDown((unsigned)in_gpio, PI_PUD_DOWN); /* avoid floating reads if unwired */

    int failures = 0;
    for (int i = 0; i < toggles; i++) {
        int level = i % 2;
        gpioWrite((unsigned)out_gpio, level);
        time_sleep(settle_s);
        int read_level = gpioRead((unsigned)in_gpio);

        int ok = (read_level == level);
        printf("toggle %2d: gpio%d=%d -> gpio%d=%d  %s\n",
               i, out_gpio, level, in_gpio, read_level, ok ? "PASS" : "FAIL");
        if (!ok) failures++;
    }

    gpioWrite((unsigned)out_gpio, 0);
    gpioTerminate();

    if (failures == 0) {
        printf("RESULT: gpio%d and gpio%d are connected (%d/%d toggles matched)\n",
               out_gpio, in_gpio, toggles, toggles);
        return 0;
    } else {
        printf("RESULT: gpio%d and gpio%d are NOT reliably connected (%d/%d toggles failed)\n",
               out_gpio, in_gpio, failures, toggles);
        return 1;
    }
}
