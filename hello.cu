#include <stdio.h>
#include <cstdint>
#include <fstream>
#include "bitmap.h"


#define COLORS_PER_PIXEL 3
#define DEBUG 0

using namespace std;

struct pixel {
  uint32_t r;
  uint32_t g;
  uint32_t b;
};

__global__
void cuda_gamma(uint8_t *pSrc, uint8_t *pDst, uint16_t imageSize) {
  size_t row = blockIdx.x * blockDim.x + threadIdx.x;
  size_t col = blockIdx.y * blockDim.y + threadIdx.y;

  size_t currentPixel = (row * COLORS_PER_PIXEL * imageSize) + (col * COLORS_PER_PIXEL);

  pDst[currentPixel]   = uint8_t(255.0f * pow(float(pSrc[currentPixel])/255.0f,   (1.0f/2.2f)));
  pDst[currentPixel+1] = uint8_t(255.0f * pow(float(pSrc[currentPixel+1])/255.0f, (1.0f/2.2f)));
  pDst[currentPixel+2] = uint8_t(255.0f * pow(float(pSrc[currentPixel+2])/255.0f, (1.0f/2.2f)));
}

__global__
void cuda_degamma(uint8_t *pSrc, uint8_t *pDst, uint16_t imageSize) {
  size_t row = blockIdx.x * blockDim.x + threadIdx.x;
  size_t col = blockIdx.y * blockDim.y + threadIdx.y;

  size_t currentPixel = (row * COLORS_PER_PIXEL * imageSize) + (col * COLORS_PER_PIXEL);

  pDst[currentPixel]   = uint8_t(255.0f * pow(float(pSrc[currentPixel])/255.0f,   2.2f));
  pDst[currentPixel+1] = uint8_t(255.0f * pow(float(pSrc[currentPixel+1])/255.0f, 2.2f));
  pDst[currentPixel+2] = uint8_t(255.0f * pow(float(pSrc[currentPixel+2])/255.0f, 2.2f));
}

__global__
void cuda_blur(uint8_t* pSrc, uint8_t* pDst, uint16_t kernelSize, int32_t imageSize) {
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
      rowMirrorOffset = -i - row + abs(row + i) ;
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

int main(int argc, char *argv[]) {
  char *imageFilename;
  uint8_t *pDevSrcImage = nullptr;
  uint8_t *pDevDstImage = nullptr;
  uint8_t *pDevGammaDstImage = nullptr;
  uint8_t *pDevDeGammaDstImage = nullptr;
  uint8_t *pHostDstImage = nullptr;
  int16_t kernelSize = 5;

  assert(argc > 1);

  if (argc > 1) {
    imageFilename = argv[1];
  }
  printf("importing %s...\n", imageFilename);

  if (argc > 2) {
    kernelSize = stoi(argv[2]);
    assert(kernelSize % 2);
  }
  printf("running with kernel size %d\n", kernelSize);

  // bring in bitmap
  Bitmap bitmap_h(imageFilename);
  bitmap_h.printBitmapInfo();

  size_t pMemSize;
  cudaDeviceGetLimit(&pMemSize, cudaLimitStackSize);
  printf("stack limit is %zd bytes\n", pMemSize);

  // copy bitmap to device memory
  printf("Copying %s to device with a size of 0x%zX bytes\n", imageFilename, bitmap_h.getImageSize());
  cudaMalloc((void **)&pDevSrcImage, bitmap_h.getImageSize() * sizeof(pDevSrcImage));
  cudaMemcpy(pDevSrcImage, bitmap_h.getStartOfImageData(), bitmap_h.getImageSize(), cudaMemcpyHostToDevice);

  // create destination/processed buffers
  cudaMalloc((void **)&pDevDstImage,        bitmap_h.getImageSize() * sizeof(pDevDstImage));
  cudaMalloc((void **)&pDevGammaDstImage,   bitmap_h.getImageSize() * sizeof(pDevGammaDstImage));
  cudaMalloc((void **)&pDevDeGammaDstImage, bitmap_h.getImageSize() * sizeof(pDevDeGammaDstImage));

  // call kernels
  dim3 blockSize = dim3(bitmap_h.getRows(),bitmap_h.getCols(),1);
  cuda_degamma<<<blockSize, 1>>>(pDevSrcImage, pDevGammaDstImage, bitmap_h.getRows());
  cuda_blur<<<blockSize, 1>>>(pDevGammaDstImage, pDevDeGammaDstImage, kernelSize, bitmap_h.getRows());
  cuda_gamma<<<blockSize, 1>>>(pDevDeGammaDstImage, pDevDstImage, bitmap_h.getRows());
  cudaDeviceSynchronize();

  // create host buffer for bmp, copy device contents back to host
  pHostDstImage = new uint8_t[bitmap_h.getImageSize()];
  cudaMemcpy(pHostDstImage, pDevDstImage, bitmap_h.getImageSize(), cudaMemcpyDeviceToHost);

  if (DEBUG) {
    for (int i=0; i<32; i++) {
      printf("pixel processed %02d: %x\n", i, pHostDstImage[2500+i]);
    }

    for (int i=0; i<32; i++) {
      printf("pixel original %02d: %x\n", i, bitmap_h.getImageBuffer()[2500+i]);
    }
  }

  // create output bmp and write out processed image
  std::ofstream outputFile;
  outputFile.open("lenna_processed.bmp");
  // write bitmap header, image
  outputFile.write(bitmap_h.getImageBuffer(), bitmap_h.getHeaderSize());
  outputFile.write((char *)pHostDstImage, bitmap_h.getImageSize());
  outputFile.close();

  cudaFree(pDevSrcImage);
  cudaFree(pDevDstImage);
  cudaFree(pDevGammaDstImage);
  cudaFree(pDevDeGammaDstImage);
  cudaDeviceReset();
  free(pHostDstImage);

  return 0;
}