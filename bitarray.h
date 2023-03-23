
/*
* Copyright (c) 2022 The University of Manchester
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program.  If not, see http://www.gnu.org/licenses/.
*/

#ifndef C_2015_bitarray_h
#define C_2015_bitarray_h

/*
	 These macros were adapted from the code initially published at the following website:
	 http://c-faq.com/misc/bitsets.html
*/

#include <limits.h>		/* for CHAR_BIT */
#define BIT_ARRAY_BITSIZE 32     
#define BIT_ARRAY_SHIFT 5        
#define BIT_ARRAY_TYPE uint32_t

#define BITMASK(	b	) 		( 1 << ((b) % BIT_ARRAY_BITSIZE ))
#define BITSLOT(	b	) 		((b) >> BIT_ARRAY_SHIFT ) // ((b) / BIT_ARRAY_BITSIZE ) //

#define BITSET(	a, b	) 	((a)[BITSLOT(b)] |=  BITMASK(b))
#define BITCLEAR(	a, b	) 	((a)[BITSLOT(b)] &= ~BITMASK(b))
#define BITTEST(	a, b	) 	((a)[BITSLOT(b)] &   BITMASK(b))

#define BITNSLOTS(	nb	) 	((nb + BIT_ARRAY_BITSIZE - 1) / BIT_ARRAY_BITSIZE)

#endif
