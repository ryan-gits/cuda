#include <stdio.h>
#include <vector>
#include <tuple>
#include <cstdint>
#include "bitmap.h"
#include <fstream>

#define BYTES_PER_PIXEL 3
#define KERNEL_SIZE 13
#define PROCESS_IMAGE 1
#define DEBUG 1

// #undef __noinline__
// #include <Magick++.h>
// #define __noinline__ __attribute__((noinline))

using namespace std;

struct pixel {
  uint8_t r;
  uint8_t g;
  uint8_t b;
};

struct pixel_yuv {
  int16_t y;
  int16_t u;
  int16_t v;
};

__global__
void cuda_blur(uint8_t* pSrc, uint8_t* pDst) {
  size_t row = blockIdx.x * blockDim.x + threadIdx.x;
  size_t col = blockIdx.y * blockDim.y + threadIdx.y;

  pixel kernel[KERNEL_SIZE*KERNEL_SIZE];
  pixel_yuv kernel_yuv[KERNEL_SIZE*KERNEL_SIZE];

  bool borderPixel = (row < KERNEL_SIZE/2       || col < KERNEL_SIZE/2 ||
                      row > 511 - KERNEL_SIZE/2 || col > 511 - KERNEL_SIZE/2);

  size_t startPos = (row * BYTES_PER_PIXEL * 512) + (col * BYTES_PER_PIXEL);

  size_t centerPixel = (KERNEL_SIZE*KERNEL_SIZE)/2;
  size_t rowStride = BYTES_PER_PIXEL * 512;

  // ignore borders for now
  if (!borderPixel) {
    // populate NxN kernel, center pixel is middle of array
    // data organized in memory from bottom left, left to right, bottom to top
    // each pixel is 3 bytes, incrementing memory order: [0] = b, [1] = g, [2] = r
    for (int i = -(KERNEL_SIZE/2); i<=KERNEL_SIZE/2; i++) {
      for (int y = -(KERNEL_SIZE/2); y<=KERNEL_SIZE/2; y++) {
        kernel[(i*KERNEL_SIZE+KERNEL_SIZE/2) + (y+KERNEL_SIZE/2)].b = pSrc[startPos   + i*rowStride + y*BYTES_PER_PIXEL];
        kernel[(i*KERNEL_SIZE+KERNEL_SIZE/2) + (y+KERNEL_SIZE/2)].g = pSrc[startPos+1 + i*rowStride + y*BYTES_PER_PIXEL];
        kernel[(i*KERNEL_SIZE+KERNEL_SIZE/2) + (y+KERNEL_SIZE/2)].r = pSrc[startPos+2 + i*rowStride + y*BYTES_PER_PIXEL];

        if (row == 256 && col == 256) {
          printf("%d %d\n", i, y);
        }
      }
      
    }

    // rgb2yuv
    for (int i=0; i<KERNEL_SIZE*KERNEL_SIZE; i++) {
      kernel_yuv[i].y = ((kernel[i].r *  66 + kernel[i].g * 129 + kernel[i].b *  25 + 128) >> 8) + 16;
      kernel_yuv[i].u = ((kernel[i].r * -38 + kernel[i].g * -74 + kernel[i].b * 112 + 128) >> 8) + 128;
      kernel_yuv[i].v = ((kernel[i].r * 112 + kernel[i].g * -94 + kernel[i].b * -18 + 128) >> 8) + 128;
    }

    // sum kernal pixels
    float ySum = 0.0f, uSum = 0.0f, vSum = 0.0f;
    float yAvg = 0.0f, uAvg = 0.0f, vAvg = 0.0f;
    for (int i=0; i<(KERNEL_SIZE*KERNEL_SIZE); i++) {
      ySum += kernel_yuv[i].y;
      uSum += kernel_yuv[i].u;
      vSum += kernel_yuv[i].v;

      if (row == 256 && col == 256 && DEBUG) {
        printf("yuv kernel %d, %03f %03f %03f\n", i, kernel_yuv[i].y, kernel_yuv[i].u, kernel_yuv[i].v);
      }
    }

    if (row == 256 && col == 256 && DEBUG) {
      printf("sum %f %f %f\n", ySum, uSum, vSum);
    }

    // average sums
    yAvg = ySum/(KERNEL_SIZE * KERNEL_SIZE);
    uAvg = uSum/(KERNEL_SIZE * KERNEL_SIZE);
    vAvg = vSum/(KERNEL_SIZE * KERNEL_SIZE);

    if (row == 256 && col == 256 && DEBUG) {
      printf("%f %f %f\n", yAvg, uAvg, vAvg);
    }

    // yuv2rgb
    int32_t c = yAvg - 16;
    int32_t d = uAvg - 128;
    int32_t e = vAvg - 128;
    kernel[centerPixel].r = max(min(((298 * c +   0 * d + 409 * e + 128) >> 8), 255), 0);
    kernel[centerPixel].g = max(min(((298 * c - 100 * d - 208 * e + 128) >> 8), 255), 0);
    kernel[centerPixel].b = max(min(((298 * c + 516 * d +   0 * e + 128) >> 8), 255), 0);
  } else {
    kernel[centerPixel].b = pSrc[startPos];
    kernel[centerPixel].g = pSrc[startPos+1];
    kernel[centerPixel].r = pSrc[startPos+2];
  }

  // populate destination buffer
  pDst[startPos]   = kernel[centerPixel].b;
  pDst[startPos+1] = kernel[centerPixel].g;
  pDst[startPos+2] = kernel[centerPixel].r;
}


int main() {
  char imageFilename[] = "lenna.bmp";
  uint8_t *pDevSrcImage = nullptr;
  uint8_t *pDevDstImage = nullptr;
  uint8_t *pHostDstImage = nullptr;
  vector<tuple<uint8_t, uint8_t, uint8_t>> pixel;

  // bring in bitmap
  Bitmap bitmap_h(imageFilename);
  bitmap_h.printBitmapInfo();

  // copy bitmap to device memory
  printf("Copying %s to device with a size of 0x%X bytes\n", imageFilename, bitmap_h.getImageSize());
  cudaMalloc((void**)&pDevSrcImage, bitmap_h.getImageSize());
  cudaMemcpy(pDevSrcImage, bitmap_h.getStartOfImageData(), bitmap_h.getImageSize(), cudaMemcpyHostToDevice);

  // create destination/processed buffer
  cudaMalloc((void**)&pDevDstImage, bitmap_h.getImageSize());

  // call kernel
  dim3 blockSize = dim3(512,512,1);
  cuda_blur<<<blockSize, 1>>>(pDevSrcImage, pDevDstImage);
  cudaDeviceSynchronize();

  printf("kernel size %d\n", KERNEL_SIZE);
  printf("kernel size/2 %d\n", KERNEL_SIZE/2);  

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
  free(pHostDstImage);
  
  return 0;
}