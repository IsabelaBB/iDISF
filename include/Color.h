/**
* Color related operations
* 
* @date September, 2019
* @note Based on the implementation and theory presented in
*       http://www.easyrgb.com/en/math.php
*       http://www.brucelindbloom.com/index.html?Math.html
*       https://en.wikipedia.org/wiki/SRGB
*       https://en.wikipedia.org/wiki/CIELAB_color_space
*/
#ifndef COLOR_H
#define COLOR_H

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Includes
//=============================================================================
#include "Utils.h"
#include <math.h>

//=============================================================================
// Constants
//=============================================================================
// Whitepoint reference
static const float D65_WHITE[] = {0.950456, 1.0, 1.088754}; // Input vals are sRGB

//=============================================================================
// Float functions
//=============================================================================
/**
* Apply the gamma correction function. The value must be sRGB normalized in [0,1].
* See the links for more details.
*/
float gammaCorr(float value);

/**
* Apply the f function. The value must be XYZ normalized by the D65 whitepoint.
* See the links for more details.
*/
float labFunc(float value);

//=============================================================================
// Float* functions
//=============================================================================
/**
* Converts a grayscale color feature vector, into a LAB one with respect to the 
* normalization value given. See the links for more details.
*/
float *convertGrayToLab(int* gray, int normval);

/**
* Converts a sRGB color feature vector, into a LAB one with respect to the 
* normalization value given. See the links for more details.
*/
float *convertsRGBToLab(int* srgb, int normval);


#ifdef __cplusplus
}
#endif

#endif
