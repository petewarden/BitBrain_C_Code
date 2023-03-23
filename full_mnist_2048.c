/*
* Copyright (c) 2022 The University of Manchester
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program.  If not, see http://www.gnu.org/licenses/.
*/

/*
 
     Needs access to working 'bitarray.h' and the files which contain the MNIST data, the ADs and their thresholds
    
     Build on most POSIX platforms with:
     'clang-8 full_mnist_2048.c -O3 -march=native -fopenmp -lm -lomp -o bb'
    
     Turn off OpenMP by commenting out the #define if not available on your platform.  Runtime will be longer
     and you will lose precise timings.
 
*/

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#include "bitarray.h" // for SBC memories

// #define USE_OPENMP // comment out if OpenMP not available
#ifdef USE_OPENMP
#include <omp.h>
#endif

#define FOR_LOOP( i, n ) for( (i) = 0; (i) < (n); (i)++ )

#define TRUE  1
#define FALSE 0

// Make vector on the heap with defined type: sz = length
#define HEAP_VEC( name, TYPE, sz ) TYPE *name; name = (TYPE *)calloc( (size_t)(sz), sizeof( TYPE ) ); if(!name) { printf(" allocation failure in heap_vector() "); exit(1); } 
#define FREE_VEC( name ) free( (void*)name );  // free vector memory

// Make matrix on the heap with defined type: szr = number of rows, szc = number of columns
#define HEAP_MAT( name, TYPE, szr, szc ) TYPE **name; name = (TYPE **) calloc( (size_t)( szr ), sizeof( TYPE* ) ); if( !name ) { printf(" allocation failure 1 in heap_matrix() "); exit(1); } name[0] = (TYPE *) calloc( (size_t)( szr * szc ), sizeof( TYPE ) ); if ( !name[0] ) { printf(" allocation failure 2 in heap_matrix() "); exit(1); } for( uint32_t i = 1; i < (szr); i++ ) name[i] = name[i-1] + szc;
#define FREE_MAT( name ) free( (void*)name[0] ); free( (void*)name );  // free matrix memory

// 3-Tensor type: szr = number of rows, szc = number of columns, szd = depth
#define HEAP_TENS( name, TYPE, szr, szc, szd ) TYPE ***name; name = (TYPE ***) calloc( (size_t)( szr ), sizeof( TYPE** ) ); if ( !name ) { printf(" allocation failure 1 in heap_tensor() "); exit(1); } else printf(" *** at %p ", &name ); name[0] = (TYPE **) calloc( (size_t)( szr * szc ), sizeof( TYPE* ) ); if ( !name[0] ) { printf(" allocation failure 2 in heap_tensor() "); exit(1); } else printf(" ** at %p ", &(name[0]) ); name[0][0] = (TYPE *) calloc( (size_t)( szr * szc * szd ), sizeof( TYPE ) ); if( !name[0][0] ) { printf(" allocation failure 3 in heap_tensor() "); exit(1); } else printf(" * at %p with %lu bytes \n", &(name[0][0]), ( szr * szc * szd ) * sizeof( TYPE ) ); for( uint32_t j = 1; j < (szc); j++ ) name[0][j] = name[0][j-1] + szd; for( uint32_t i = 1; i < (szr); i++ ) { name[i] = name[i-1] + szc; name[i][0] = name[i-1][0] + szc * szd; for( j = 1; j < (szc); j++ ) name[i][j] = name[i][j-1] + szd; }
#define FREE_TENS( name ) free( (void*) name[0][0] ); free( (void*) name[0] ); free( (void*) name );

// Make bit array on the heap with defined type
#define MAKE_BITARRAY( name, sz ) BIT_ARRAY_TYPE * name; name = calloc( BITNSLOTS( sz ), sizeof( BIT_ARRAY_TYPE ) ); if( name == NULL ) { printf(" %s could not be allocated \n", #name ); exit(1); }

uint64_t ws, ds, wd, ww, wwd; // used for memory calculations, need to be globals

#define MAT_INDEX_2D( i, j, s1 )          ((i)*(s1) + (j))              // for accessing 2D bit matrix 
#define TENS_INDEX_3D( i, j, l, s1, s2 )  ((i) + (j)*(s1) + (l)*(s2))   // for accessing 3D bit tensor  
// for cases here  ws = W  wd = W * D
#define D2( i, j ) 			((i)*ws + (j))
#define D3( i, j, k ) 		((i) + (j)*(wd) + (k)*ws)


/*
	================================================
	Return vector of ADEs that fire in ad_row_fired
	================================================
	
	Most basic case with no Hebbian learning or noise added to pixels. Assumes all ADEs have same width per AD.
	
	Inputs:
	-------
	loop - position in training or test set (i.e. 60k elements in MNIST training, 10k elements in MNIST test)
	
	w - length of this AD
	
	n - number of synapses in ADE (for this AD)
	
	sensory_input - data length-by-data width pre-loaded matrix of input values (i.e. 0.255 for E/MNIST)
	
	ad_row_thresh - w vector of firing thresholds per ADE
	
	address_decoder - w-by-n matrix of sample positions from input data vector (i.e. 784 possible elements for E/MNIST).
	Importantly; positive values are excitatory synapses, negative values are inhibitory synapses.
	
	Return:
	-------
	ad_row_fired - w vector relating ADEs that fired = 1 and didn't fire = 0
	
*/
void	find_AD_firing_pattern( uint32_t loop, uint32_t w, uint8_t n, uint8_t** sensory_input, int32_t* ad_row_thresh, int32_t** address_decoder, uint8_t* ad_row_fired )
{
	uint32_t	i, j, total = 0;
	int32_t 	count, position;
	int8_t 	yang;
	int16_t  pixel;

	FOR_LOOP( i, w ) {  // loop over all ADEs in AD
		
		count = 0;
		
		FOR_LOOP( j, n ) {  	// loop over all synapses in this ADE
		
		   if( address_decoder[i][j] > 0 ) 
		      { yang =  64; position =  address_decoder[i][j]; }
		   else  
		      { yang = -64; position = -address_decoder[i][j]; }
		   
		   pixel = sensory_input[ loop ][ position-1 ];

		   count += ( pixel - 127 ) * yang;
		   
			}  // j over N
      
		ad_row_fired[ i ] = ( count >= ad_row_thresh[ i ] ? TRUE : FALSE );
		
		}  // i over W
}


// write bits to SBC for supervised learning
uint32_t write_to_sbc( uint8_t* first, uint8_t* second, BIT_ARRAY_TYPE* mem, uint32_t w, uint8_t label )
{
	uint32_t i, j, internal_count = 0;
	uint64_t tens_index;

	FOR_LOOP( i, w ) {
		if( first[i] ) {
			
			FOR_LOOP( j, w )
				if( second[j] ) {
					
					tens_index = D3( i, j, label );
						
					if( !BITTEST( mem, tens_index ) ) {

						internal_count++;  

						BITSET( mem, tens_index );
						}
					}
			}
		}
		
	return internal_count;
}


// read bits from SBC for inference
void read_from_sbc( uint8_t* first, uint8_t* second, BIT_ARRAY_TYPE* mem, uint32_t w, uint32_t* store )
{
	uint32_t i, j, k;
	uint64_t tens_index;
	uint8_t 	bit_test;

	FOR_LOOP( i, w )
		if( first[i] )
			
			FOR_LOOP( j, w )
				if( second [j] ) {

					FOR_LOOP( k, ds ) {
					
						tens_index = D3( i, j, k );
						
						bit_test = BITTEST( mem, tens_index ); 

                  if( bit_test ) (store[k])++;
//						store[k] += ( bit_test != 0 );
						}
						
					}
}


//	load int32_t matrix with error checking
uint8_t load_int32_mat_from_file( const char *file_name, int32_t** into, size_t rows, size_t cols )
{
	FILE 		*infile_ptr;
	size_t 	total_read, expected_sz = rows * cols;
	
// open file
	if ( (infile_ptr = fopen( file_name, "rb")) == NULL ) {
		printf("\n Couldn't open %s. \n", file_name );
		printf(" file error 1 in load_int32_mat_from_file()");
		return FALSE;
		}
	
// read in whole matrix in one big blob and check for errors
	total_read = fread( &into[0][0], sizeof(int32_t), expected_sz, infile_ptr );

	if( total_read != expected_sz || total_read == 0 || total_read == EOF )
		printf("\n Potential problem with matrix read in load_int32_mat_from_file() - BEWARE \n");
	
	fclose( infile_ptr );
	
	return TRUE;
}


// load int32_t vector with error checking
uint8_t load_int32_vec_from_file( const char *file_name, int32_t* into, size_t size )
{
	FILE 		*infile_ptr;
	size_t 	total_read;
	
// open file
	if ( (infile_ptr = fopen( file_name, "rb")) == NULL ) {
		printf("\n Couldn't open %s. \n", file_name );
		printf(" file error 1 in load_int32_vec_from_file()");
		return FALSE;
		}
	
// read in whole vector in one big blob and check for errors
	total_read = fread( &into[0], sizeof(int32_t), size, infile_ptr );

	if( total_read != size || total_read == 0 || total_read == EOF )
		printf("\n Potential problem with vector read in load_int32_vec_from_file() - BEWARE \n");
	
	fclose( infile_ptr );
	
	return TRUE;
}


// load uint8_t matrix with error checking
uint8_t load_uint8_mat_from_file( const char *file_name, uint8_t** into, size_t rows, size_t cols )
{
	FILE 		*infile_ptr;
	size_t 	total_read, expected_sz = rows * cols;
	
// open file
	if ( (infile_ptr = fopen( file_name, "rb")) == NULL ) {
		printf("\n Couldn't open %s. \n", file_name );
		printf(" file error 1 in load_uint8_mat_from_file()");
		return FALSE;
		}
	
// read in whole matrix in one big blob and check for errors
	total_read = fread( &into[0][0], sizeof(uint8_t), expected_sz, infile_ptr );

	if( total_read != expected_sz || total_read == 0 || total_read == EOF )
		printf("\n Potential problem with matrix read in load_uint8_mat_from_file() - BEWARE \n");
	
	fclose( infile_ptr );
	
	return TRUE;
}


// load uint8_t vector with error checking
uint8_t load_uint8_vec_from_file( const char *file_name, uint8_t* into, size_t size )
{
	FILE 		*infile_ptr;
	size_t 	total_read;
	
// open file
	if ( (infile_ptr = fopen( file_name, "rb")) == NULL ) {
		printf("\n Couldn't open %s. \n", file_name );
		printf(" file error 1 in load_uint8_vec_from_file()");
		return FALSE;
		}
	
// read in whole vector in one big blob and check for errors
	total_read = fread( &into[0], sizeof(uint8_t), size, infile_ptr );

	if( total_read != size || total_read == 0 || total_read == EOF )
		printf("\n Potential problem with vector read in load_uint8_vec_from_file() - BEWARE \n");
	
	fclose( infile_ptr );
	
	return TRUE;
}


// show firing pattern of the AD from a given input
void show_firing_pattern( uint8_t* ad_fire, uint16_t sz )
{
   uint16_t i;

   FOR_LOOP( i, sz )
      printf( "%1u ", ad_fire[i] );
   
   printf("\n");
}


// show thresholds in the AD
void show_thresholds( int32_t* ad_thresh, uint16_t sz )
{
   uint16_t i;

   printf("\n");
   
   FOR_LOOP( i, sz )
      printf( "%d ", ad_thresh[i] );
   
   printf("\n");
}


// show an AD synapse sampling pattern
void show_AD( int32_t** ad, uint16_t w, uint8_t n )
{
   uint16_t i, j;

   printf("\n");
   
   FOR_LOOP( i, w ) {
   
      FOR_LOOP( j, n )
         printf( "%5d ", ad[i][j] );
      
      printf("\n");
      }
   
   printf("\n");
}


// test program
int main( int argc, char** argv )
{
   uint32_t i, j, k, wrong = 0;
   uint8_t  label, inf_label;
   double   pct;

#ifdef USE_OPENMP
   double 	before, after;
   #define  BEFORE before = omp_get_wtime(); 
   #define  AFTER( X )  after = omp_get_wtime(); printf("\n %s = %8.4f secs \n", X, after - before );
   printf("\n Threads available = %3u \n", omp_get_max_threads() );
#else
   #define BEFORE 
   #define AFTER( X )  
#endif

// define problem sizes - files are for MNIST and these sizes at the moment
   #define W 2048  
   #define D 10  

   #define N1	6 
   #define N2	8 
   #define N3	10 
   #define N4	12 
   
   #define TRAIN_SZ 60000
   #define TEST_SZ  10000
   #define INPUT_SZ 784

   #define SBC_CT 6  

   ws = W, ds = D, wd = W * D, ww = W * W, wwd = W * W * D;  

BEFORE
// allocate memory on the heap safely for essential vectors and matrices

   HEAP_VEC( AD_fire_1, uint8_t, W )  // AD firing patterns
   HEAP_VEC( AD_fire_2, uint8_t, W )
   HEAP_VEC( AD_fire_3, uint8_t, W )
   HEAP_VEC( AD_fire_4, uint8_t, W )

   HEAP_MAT( training_data, uint8_t, TRAIN_SZ, INPUT_SZ ) // matrices holding MNIST data
   HEAP_MAT( test_data,     uint8_t, TEST_SZ,  INPUT_SZ )
   
   HEAP_VEC( train_label, uint8_t, TRAIN_SZ )  // vectors holding data labels
   HEAP_VEC( test_label,  uint8_t, TEST_SZ )

   HEAP_VEC( thresh1,  int32_t, W ) // thresholds from unsupervised learning
   HEAP_VEC( thresh2,  int32_t, W )
   HEAP_VEC( thresh3,  int32_t, W )
   HEAP_VEC( thresh4,  int32_t, W )

   HEAP_MAT( AD1, int32_t, W, N1 ) // ADs from unsupervised learning
   HEAP_MAT( AD2, int32_t, W, N2 )
   HEAP_MAT( AD3, int32_t, W, N3 )
   HEAP_MAT( AD4, int32_t, W, N4 )
   
   MAKE_BITARRAY( D3_bmatrix12, wwd ) // SBC memories
   MAKE_BITARRAY( D3_bmatrix13, wwd )
   MAKE_BITARRAY( D3_bmatrix14, wwd )
   MAKE_BITARRAY( D3_bmatrix23, wwd )
   MAKE_BITARRAY( D3_bmatrix24, wwd )
   MAKE_BITARRAY( D3_bmatrix34, wwd )
   
   HEAP_MAT( class_count,  uint32_t, SBC_CT, D ) // for collecting & summing class bits
   HEAP_VEC( class_vector, uint32_t, D ) 

   HEAP_MAT( confusion,  uint32_t, D, D ) // for confusion matrix

AFTER( "Memory allocations" )   
BEFORE


// load files into data structures
   load_uint8_mat_from_file( "train_data", training_data, TRAIN_SZ, INPUT_SZ );
   load_uint8_mat_from_file( "test_data",  test_data,     TEST_SZ,  INPUT_SZ );
   
   load_uint8_vec_from_file( "train_label", train_label, TRAIN_SZ );
   load_uint8_vec_from_file( "test_label",  test_label,  TEST_SZ );
   
   load_int32_mat_from_file( "AD1_2048", AD1, W, N1 );
   load_int32_mat_from_file( "AD2_2048", AD2, W, N2 );
   load_int32_mat_from_file( "AD3_2048", AD3, W, N3 );
   load_int32_mat_from_file( "AD4_2048", AD4, W, N4 );

   load_int32_vec_from_file( "thresh1_2048", thresh1, W );
   load_int32_vec_from_file( "thresh2_2048", thresh2, W );
   load_int32_vec_from_file( "thresh3_2048", thresh3, W );
   load_int32_vec_from_file( "thresh4_2048", thresh4, W );

AFTER( "File loads" ) 
// confirm files loaded correctly by showing them
//#define SHOW_LOADED_FILES  
#ifdef SHOW_LOADED_FILES
   show_thresholds( thresh1, W );
   show_thresholds( thresh2, W );
   show_thresholds( thresh3, W );
   show_thresholds( thresh4, W );

   show_AD( AD1, W, N1 );
   show_AD( AD2, W, N2 );
   show_AD( AD3, W, N3 );
   show_AD( AD4, W, N4 );
#endif


BEFORE
// supervised learning on training set
   FOR_LOOP( i, TRAIN_SZ ) {
    
// find AD firing patterns         
#pragma omp parallel sections shared( training_data, i ) num_threads( 4 )
{
#pragma omp section
      find_AD_firing_pattern( i, W, N1, training_data, thresh1, AD1, AD_fire_1 );
#pragma omp section
      find_AD_firing_pattern( i, W, N2, training_data, thresh2, AD2, AD_fire_2 );
#pragma omp section
      find_AD_firing_pattern( i, W, N3, training_data, thresh3, AD3, AD_fire_3 );
#pragma omp section
      find_AD_firing_pattern( i, W, N4, training_data, thresh4, AD4, AD_fire_4 );
}

// show every X firing patterns from training set
//#define SHOW_FIRING_PATTERNS 10000 
#ifdef SHOW_FIRING_PATTERNS
      if( i % SHOW_FIRING_PATTERNS == 0 ) {
         show_firing_pattern( AD_fire_1, W );
         show_firing_pattern( AD_fire_2, W );
         show_firing_pattern( AD_fire_3, W );
         show_firing_pattern( AD_fire_4, W );
         printf("\n\n");
         }  
#endif

      label = train_label[i]; // load label for this input

// write coincidences into SBC memories
#pragma omp parallel sections num_threads( 6 ) shared( label )
{
#pragma omp section
      write_to_sbc( AD_fire_1, AD_fire_2, D3_bmatrix12, W, label );
#pragma omp section
      write_to_sbc( AD_fire_2, AD_fire_4, D3_bmatrix24, W, label );
#pragma omp section
      write_to_sbc( AD_fire_3, AD_fire_4, D3_bmatrix34, W, label );
#pragma omp section
      write_to_sbc( AD_fire_1, AD_fire_3, D3_bmatrix13, W, label );
#pragma omp section
      write_to_sbc( AD_fire_1, AD_fire_4, D3_bmatrix14, W, label  );
#pragma omp section
      write_to_sbc( AD_fire_2, AD_fire_3, D3_bmatrix23, W, label );
}
      
      }
AFTER( "Supervised learning" ) 


BEFORE
// inference on test set
   FOR_LOOP( i, TEST_SZ ) {

// clear bit counters   
      FOR_LOOP( j, SBC_CT ) 
         FOR_LOOP( k, D )
            class_count[j][k] = 0;

      FOR_LOOP( j, D ) 
         class_vector[j] = 0;

// find AD firing patterns         
#pragma omp parallel sections shared( test_data, i ) num_threads( 4 )
{
#pragma omp section
      find_AD_firing_pattern( i, W, N1, test_data, thresh1, AD1, AD_fire_1 );
#pragma omp section
      find_AD_firing_pattern( i, W, N2, test_data, thresh2, AD2, AD_fire_2 );
#pragma omp section
      find_AD_firing_pattern( i, W, N3, test_data, thresh3, AD3, AD_fire_3 );
#pragma omp section
      find_AD_firing_pattern( i, W, N4, test_data, thresh4, AD4, AD_fire_4 );
}

// read coincidences from SBC memories
#pragma omp parallel sections num_threads( 6 )  shared( class_count )
{
#pragma omp section
      read_from_sbc( AD_fire_1, AD_fire_2, D3_bmatrix12, W, class_count[0] );
#pragma omp section
      read_from_sbc( AD_fire_2, AD_fire_4, D3_bmatrix24, W, class_count[1] );
#pragma omp section
      read_from_sbc( AD_fire_3, AD_fire_4, D3_bmatrix34, W, class_count[2] );
#pragma omp section
      read_from_sbc( AD_fire_1, AD_fire_3, D3_bmatrix13, W, class_count[3] );
#pragma omp section
      read_from_sbc( AD_fire_1, AD_fire_4, D3_bmatrix14, W, class_count[4]  );
#pragma omp section
      read_from_sbc( AD_fire_2, AD_fire_3, D3_bmatrix23, W, class_count[5] );
}

// collect counts
      FOR_LOOP( j, D )
         FOR_LOOP( k, SBC_CT )
            class_vector[j] += class_count[k][j];

// find label of highest count   
      inf_label = 0;
      k = 0;
      FOR_LOOP( j, D )
         if( class_vector[j] > k ) {
            k = class_vector[j];
            inf_label = j;
            }

// if not correct record it     
      label = test_label[i];
      if( inf_label != label ) 
         wrong++;

// build confusion matrix         
      ( confusion[ label ][ inf_label ] )++;
      
      }
AFTER( "Inference" ) 


// print performance
   printf("\n %u wrong = %6.3f pct correct \n", wrong, 100.0 - wrong / ( TEST_SZ / 100.0 ) );

// show confusion matrix   
   printf("\n");
   FOR_LOOP( j, D ) {
      FOR_LOOP( k, D )
         printf( "%5u ", confusion[j][k] );
      printf("\n");
      }
   printf("\n");

   return 0;
}
