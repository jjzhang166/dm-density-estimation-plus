/* Copyright (c) 2010, Simeon Bird <spb41@cam.ac.uk>
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE. */
#ifndef __READ_UTILS_H
#define __READ_UTILS_H

#include <stdint.h>
/** \file 
 * Contains multi_endian_swap() and endian_swap() */

#ifdef __cplusplus
extern "C"{
#endif //__cplusplus
/*Functions to swap the enddianness of shorts and ints.*/
//inline void endian_swap(uint16_t& x);

/** Function to swap the enddianness of ints.
 * @param x int to swap*/
#if defined(_WIN32) || defined(_WIND64)
	__inline uint32_t endian_swap(uint32_t* x)
#else
	inline uint32_t endian_swap(uint32_t* x)
#endif
{
    *x = (*x>>24) | 
        ((*x<<8) & 0x00FF0000) |
        ((*x>>8) & 0x0000FF00) |
        (*x<<24);
    return *x;
}

/** Swap the endianness of a range of integers
 * @param start Pointer to memory to start at.
 * @param range Range to swap, in bytes.*/
void multi_endian_swap(uint32_t * start,int32_t range);

#ifdef __cplusplus
        }
#endif //__cplusplus
#endif //__READ_UTILS_H

