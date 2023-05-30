#include "isp.hpp"

__global__
void box_blur(uint8_t* pSrc, uint8_t* pDst, uint16_t kernelSize, int32_t imageSize) {
  int32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t col = blockIdx.y * blockDim.y + threadIdx.y;

  int32_t numPixelsKernel = kernelSize * kernelSize;
  int32_t rowStride       = COLORS_PER_PIXEL * imageSize;
  int32_t colStride       = COLORS_PER_PIXEL;
  int32_t startPos        = (row * rowStride) + (col * colStride);

  pixel currentPixel;

  float rSum = 0.0f, gSum = 0.0f, bSum = 0.0f;
  float rAvg = 0.0f, gAvg = 0.0f, bAvg = 0.0f;

  int16_t rowMirrorOffset = 0;
  int16_t colMirrorOffset = 0;

  // data organized in memory from bottom left, left to right, bottom to top
  // each pixel is 3 bytes, incrementing memory order: [0] = b, [1] = g, [2] = r
  for (int32_t i = -(kernelSize/2); i<=kernelSize/2; i++) {
    // mirror about x axis for row border pixels
    // top row mirror pixels
    if (row + i > imageSize-1) {
      rowMirrorOffset = -i - (row + i - imageSize + 1);
    // bottom row mirror pixels
    } else if (row + i < 0) {
      rowMirrorOffset = -i - row + abs(row + i) + 1;
    } else {
      rowMirrorOffset = 0;
    }

    for (int32_t y = -(kernelSize/2); y<=kernelSize/2; y++) {
      // mirror about y axis for column border pixels
      // right column mirror pixels
      if (col + y > imageSize-1) {
        colMirrorOffset = -y - (col + y - imageSize + 1);
      // left column mirror pixels
      } else if (col + y < 0) {
        colMirrorOffset = -y + abs(col + y) - 1;
      } else {
        colMirrorOffset = 0;
      }

      currentPixel.b = pSrc[startPos   + (i+rowMirrorOffset)*rowStride + (y+colMirrorOffset)*colStride];
      currentPixel.g = pSrc[startPos+1 + (i+rowMirrorOffset)*rowStride + (y+colMirrorOffset)*colStride];
      currentPixel.r = pSrc[startPos+2 + (i+rowMirrorOffset)*rowStride + (y+colMirrorOffset)*colStride];

      rSum += currentPixel.r;
      gSum += currentPixel.g;
      bSum += currentPixel.b;
    }
  }

  rAvg = rSum/numPixelsKernel;
  gAvg = gSum/numPixelsKernel;
  bAvg = bSum/numPixelsKernel;

  pDst[startPos]   = uint8_t(max(min(bAvg, 255.0f), 0.0f));
  pDst[startPos+1] = uint8_t(max(min(gAvg, 255.0f), 0.0f));
  pDst[startPos+2] = uint8_t(max(min(rAvg, 255.0f), 0.0f));
}
