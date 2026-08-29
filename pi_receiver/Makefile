CC ?= gcc
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -Werror

pmu_uart_receiver: pmu_uart_receiver.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f pmu_uart_receiver