CC := gcc
CFLAGS := -O2 -Wall -Wextra
LDLIBS := -lpigpio -lpthread -lrt -lm

.PHONY: all clean
all: transmitter receiver check_loopback loopback_combined

transmitter: transmitter.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

receiver: receiver.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

check_loopback: check_loopback.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

loopback_combined: loopback_combined.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm -f transmitter receiver check_loopback loopback_combined
