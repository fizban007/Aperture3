/////////////////////////////////////////////////////////////////////////////////////////
///
///           \file  cudaUtility.h
///
/// __Description__:     Defines useful error-checking inline functions. Included in
///                      cuda kernel implementation files.
///

/// __Version__:         1.0\n
/// __Author__:          Alex Chen, fizban007@gmail.com\n
/// __Organization__:    Columbia University
///
/////////////////////////////////////////////////////////////////////////////////////////

#ifndef  __CUDAUTILITY_INC
#define  __CUDAUTILITY_INC

#define CUDA_ERROR_CHECK        //!< Defines whether to check error
#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )  //!< Wrapper to allow display of file and line number
#define CudaCheckError() __cudaCheckError( __FILE__, __LINE__ ) //!< Wrapper to allow display of file and line number
#define EPS 1.0e-10                                             //!< Smallest floating point difference to be tolerated when checking a floating number against zero
 
#include <stdio.h>

////////////////////////////////////////////////////////////////////////////////
///  Checks last kernel launch error. 
////////////////////////////////////////////////////////////////////////////////
inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
     
    // More careful checking. However, this will affect performance.
    // Comment away if needed.
#ifndef NDEBUG
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

#endif
      
    return;
}


////////////////////////////////////////////////////////////////////////////////
///  Checks memory allocation error
////////////////////////////////////////////////////////////////////////////////
inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                file, line, cudaGetErrorString( err ) );
        cudaGetLastError();
        // exit(-1);
        throw(cudaGetErrorString(err));
    }
#endif
     
    return;
}

#endif   // ----- #ifndef __CUDAUTILITY_INC  ----- 

//vim:syntax=cuda.doxygen
