__global__
void degamma(uint8_t *pSrc, uint8_t *pDst, uint16_t imageSize) {
  int32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t col = blockIdx.y * blockDim.y + threadIdx.y;

  int32_t currentPixel = (row * 3 * imageSize) + (col * 3);

  pDst[currentPixel]   = uint8_t(255.0f * pow(float(pSrc[currentPixel])/255.0f,   2.2f));
  pDst[currentPixel+1] = uint8_t(255.0f * pow(float(pSrc[currentPixel+1])/255.0f, 2.2f));
  pDst[currentPixel+2] = uint8_t(255.0f * pow(float(pSrc[currentPixel+2])/255.0f, 2.2f));
}