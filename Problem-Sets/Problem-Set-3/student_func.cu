/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.


  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include <stdio.h>
#include "reference_calc.cpp"
#include "utils.h"

#define NB_THREADS  1024

__global__ void histo(const float* const lum,
                                  unsigned int* const d_histo,
                                  float lumRange,
                                  float lumMin,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins){

  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);

  if ( thread_2D_pos.x >= numCols ||
       thread_2D_pos.y >= numRows )
  {
      return;
  }

  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

  int bin = ((lum[thread_1D_pos] - lumMin) / lumRange) * numBins;
  atomicAdd((d_histo+bin), 1);
}


__global__ void excluScan(unsigned int* d_cdf, const size_t numRows, const size_t numCols, const size_t numBins){

    const int tid = threadIdx.x;

    extern __shared__ float s_cdf[];

    // First step of the Hillis Steel Scan, we write into shared memory.
    // Instead of having a copy-only step.
    if (tid >= 1){
        s_cdf[tid] = d_cdf[tid] + d_cdf[tid - 1];
    }

    for (int hop = 2 ; hop < numBins ; hop <<= 1){
        if (tid >= hop){
            s_cdf[tid] += s_cdf[tid - hop];
        }
        __syncthreads();
    }
    // Inclusive to exclusive scan back to global memory
    if (tid > 0){
        d_cdf[tid] = s_cdf[tid-1];
    }
    d_cdf[0] = 0;
}

__global__ void reduceMinMax(const float* const d_logLuminance,
                                float * d_outLum,
                                const size_t numRows,
                                const size_t numCols,
                                const int isMax){

    extern __shared__ float s_logLum[];

    int tid = threadIdx.x;
    int global_tid = blockIdx.x * blockDim.x + tid;

    if (global_tid >= (numRows*numCols)){
        return;
    }

    // Copy of the luminance from global memory to shared memory
    // to reduce the access time
    s_logLum[tid] = d_logLuminance[global_tid];
    __syncthreads();

    // At each iteration, we min/max 2 elements and store it in the left
    // side of the array. The width is then halved. Until we end up with only
    // 1 element at position [0]
    unsigned int i;
    for (i = blockDim.x/2 ; i > 0 ; i >>= 1){
        if (tid < i && (tid+i) < (numRows*numCols)){
            if (isMax){
                s_logLum[tid] = max(s_logLum[tid], s_logLum[tid + i]);
            } else {
                s_logLum[tid] = min(s_logLum[tid], s_logLum[tid + i]);
            }
        }
        __syncthreads();
    }

    // Output the max of the block to global memory.
    if (tid == 0){
        *(d_outLum + blockIdx.x) = s_logLum[0];
    }
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */

    int nb_blocks = ((numRows*numCols) / NB_THREADS) + 1;

    float * d_intermediateLogLum;
    checkCudaErrors(cudaMalloc(&d_intermediateLogLum, sizeof(float) * nb_blocks));

    // First call to reduceMinMax to do the MIN
    reduceMinMax<<<nb_blocks, NB_THREADS, sizeof(float) * NB_THREADS>>>(d_logLuminance, d_intermediateLogLum, numRows, numCols, 0);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    float * d_min_logLum;
    checkCudaErrors(cudaMalloc(&d_min_logLum, sizeof(float)));
    reduceMinMax<<<1, NB_THREADS, sizeof(float) * NB_THREADS>>>(d_intermediateLogLum, d_min_logLum, numRows, numCols, 0);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpy(&min_logLum, d_min_logLum, sizeof(float), cudaMemcpyDeviceToHost));

    // Now, the MAX
    reduceMinMax<<<nb_blocks, NB_THREADS, sizeof(float) * NB_THREADS>>>(d_logLuminance, d_intermediateLogLum, numRows, numCols, 1);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    float * d_max_logLum;
    checkCudaErrors(cudaMalloc(&d_max_logLum, sizeof(float)));
    reduceMinMax<<<1, NB_THREADS, sizeof(float) * NB_THREADS>>>(d_intermediateLogLum, d_max_logLum, numRows, numCols, 1);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpy(&max_logLum, d_max_logLum, sizeof(float), cudaMemcpyDeviceToHost));

    // We now can calculate the range of luminance.
    float lumRange = max_logLum - min_logLum;

    //printf("min: %f\n", min_logLum);
    //printf("max: %f\n", max_logLum);

    const dim3 blockSize(32, 32, 1);
    const dim3 gridSize(numCols/32+1, numRows/32+1, 1);

    // I now create a histogram on numBins channels.
    histo<<<gridSize, blockSize>>>(d_logLuminance, d_cdf, lumRange, min_logLum, numRows, numCols, numBins);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    // Finally a hillis and steele exclusive scan is performed.
    excluScan<<<1, numBins, sizeof(float) * numBins>>>(d_cdf, numRows, numCols, numBins);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaFree(d_intermediateLogLum));
    checkCudaErrors(cudaFree(d_min_logLum));
    checkCudaErrors(cudaFree(d_max_logLum));
}
