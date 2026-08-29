/*
 * Minimal board peripheral stub matching the SDK-generated MCXN947 example layout.
 */

#ifndef _PERIPHERALS_H_
#define _PERIPHERALS_H_

#include "fsl_common.h"

#ifdef __cplusplus
extern "C" {
#endif

void BOARD_InitPeripherals(void);
void BOARD_InitBootPeripherals(void);

#ifdef __cplusplus
}
#endif

#endif /* _PERIPHERALS_H_ */
