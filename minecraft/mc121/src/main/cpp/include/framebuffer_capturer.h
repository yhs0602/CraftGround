#pragma once
#include <jni.h>

extern jclass byteStringClass;
extern jmethodID copyFromMethod;

enum EncodingMode { RAW = 0, PNG = 1, ZEROCOPY = 2 };