#include <stdio.h>
#include <cstdint>
#include <fstream>
#include "bitmap.h"
#include "gamma.cu"
#include "degamma.cu"
#include "box_blur.cu"

#define DEBUG 0

using namespace std;

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
  degamma<<<blockSize, 1>>>(pDevSrcImage, pDevGammaDstImage, bitmap_h.getRows());
  box_blur<<<blockSize, 1>>>(pDevGammaDstImage, pDevDeGammaDstImage, kernelSize, bitmap_h.getRows());
  gamma<<<blockSize, 1>>>(pDevDeGammaDstImage, pDevDstImage, bitmap_h.getRows());
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
  outputFile.open("./resources/lenna_processed.bmp");
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