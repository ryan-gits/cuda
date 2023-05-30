#include "isp.hpp"

__global__
void gamma(uint8_t *pSrc, uint8_t *pDst, int32_t imageSize) {
  int32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t col = blockIdx.y * blockDim.y + threadIdx.y;

  int32_t currentPixel = (row * COLORS_PER_PIXEL * imageSize) + (col * COLORS_PER_PIXEL);

  pDst[currentPixel]   = uint8_t(255.0f * pow(float(pSrc[currentPixel])/255.0f,   (1.0f/2.2f)));
  pDst[currentPixel+1] = uint8_t(255.0f * pow(float(pSrc[currentPixel+1])/255.0f, (1.0f/2.2f)));
  pDst[currentPixel+2] = uint8_t(255.0f * pow(float(pSrc[currentPixel+2])/255.0f, (1.0f/2.2f)));
}