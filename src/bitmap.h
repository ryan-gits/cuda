// bitmaps include a header
// pixels start at X offset
// pixels start from bottom left corner image, left to right, bottom to top
// pixel color order: BGR in incrementing memory

#ifndef BITMAP_H_
#define BITMAP_H_

#include <stdio.h>
#include <tuple>
#include <cstdint>
#include <vector>
#include <fstream>
#include <assert.h>
#include <memory>

#define HEADER_IMAGE_START_OFFSET 10
#define HEADER_COLS_OFFSET 18
#define HEADER_ROWS_OFFSET 22
#define HEADER_BIT_DEPTH_OFFSET 28
#define BITMAP_HEADER_CONST 0x4D42

class Bitmap {
  struct bitmapType {
    uint32_t rows;
    uint32_t cols;
    uint32_t bitDepth;
    uint32_t imageOffset;
    bool defined = false;
  };

  typedef std::tuple<uint8_t, uint8_t, uint8_t> pixel;

  public:
    Bitmap(std::string filename="") {
      _inputFile.open(filename);
      assert(_inputFile.is_open() && filename != "");

      // copy file into memory
      _inputFileSize = getFileSize(_inputFile);
      _pImageBuf = std::unique_ptr<char[]>(new char[_inputFileSize]);
      _inputFile.read(_pImageBuf.get(), _inputFileSize);
      _inputFile.close();
      assert(readHeaderWord(_pImageBuf, 0, 2) == BITMAP_HEADER_CONST);

      // extract header/bitmap properties and data
      _bitmapType = extractHeaderBitmapInfo(_pImageBuf);
      _image.resize(_bitmapType.rows, std::vector<pixel>(_bitmapType.cols));
      extractImageData(_pImageBuf, _bitmapType.imageOffset, _image);
    }

    uint16_t getRows() {
      return _bitmapType.rows;
    }

    uint16_t getCols() {
      return _bitmapType.cols;
    }

    uint16_t getBitDepth() {
      return _bitmapType.bitDepth;
    }

    size_t getBufferSize() {
      return _inputFileSize;
    }
    
    size_t getHeaderSize() {
      return _bitmapType.imageOffset; 
    }

    size_t getImageSize() {
      return getBufferSize() - getImageOffset();
    }

    size_t getImageOffset() {
      return _bitmapType.imageOffset;
    }

    char* getImageBuffer() {
      return _pImageBuf.get();
    }

    char* getStartOfImageData() {
      return (char *)(_pImageBuf.get() + getImageOffset());
    }

    void printBitmapInfo(uint32_t readoutCount=0) {
      printf("image information\n");
      printf("rows = %d\n", _bitmapType.rows);
      printf("cols = %d\n", _bitmapType.cols);
      printf("bit depth = %d\n", _bitmapType.bitDepth);
      printf("image offset = %d\n", _bitmapType.imageOffset);

      for (uint32_t i=0; i<readoutCount; i++) {
        printf("%02x %02x\n", i, _pImageBuf[i]);
      }
    }

  private:
    std::vector<std::vector<pixel>> _image;
    std::ifstream _inputFile;
    std::size_t _inputFileSize;
    std::unique_ptr<char[]> _pImageBuf = nullptr;
    bitmapType _bitmapType;

    size_t getFileSize (std::ifstream &file) {
      size_t tmpSize;
      file.seekg(0, file.end);
      tmpSize = file.tellg();
      file.seekg(0, file.beg);

      return tmpSize;
    }

    bitmapType extractHeaderBitmapInfo(std::unique_ptr<char[]> &buffer) {
      bitmapType bitmapTypeTmp;
      bitmapTypeTmp.cols = readHeaderWord(buffer, HEADER_COLS_OFFSET, 4);
      bitmapTypeTmp.rows = readHeaderWord(buffer, HEADER_ROWS_OFFSET, 4);
      bitmapTypeTmp.bitDepth = readHeaderWord(buffer, HEADER_BIT_DEPTH_OFFSET, 2);
      bitmapTypeTmp.imageOffset = readHeaderWord(buffer, HEADER_IMAGE_START_OFFSET, 4);
      bitmapTypeTmp.defined = true;

      return bitmapTypeTmp;
    }

    uint32_t readHeaderWord(std::unique_ptr<char[]> &buffer, uint32_t offset, uint32_t bytes) {
      assert(bytes != 0 && bytes <= 4);

      uint32_t dataTmp = 0;
      for (uint32_t i = 0; i < bytes; i++) {
        dataTmp = dataTmp + (buffer[offset+i] << i*8);
      }

      return dataTmp;
    }

    void extractImageData(std::unique_ptr<char[]> &buffer, uint32_t offset, std::vector<std::vector<pixel>> image) {
      /*
      printf("extracting data...\n");

      for (int i=0; i<64; i++) {
        printf("%02x, %02x\n", i, buffer[i]);
      }
      */
    }
};

#endif