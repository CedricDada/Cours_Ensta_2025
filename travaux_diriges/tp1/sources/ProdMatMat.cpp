#include <algorithm>
#include <cassert>
#include <iostream>
#include <thread>
#if defined(_OPENMP)
#include <omp.h>
#endif
#include "ProdMatMat.hpp"

        namespace {
            void prodSubBlocks(int iRowBlkA, int iColBlkB, int iColBlkA, int szBlock,
                               const Matrix& A, const Matrix& B, Matrix& C) {
                for (int j = iColBlkB; j < std::min(B.nbCols, iColBlkB + szBlock); ++j) {
                    for (int k = iColBlkA; k < std::min(A.nbCols, iColBlkA + szBlock); ++k) {
                        const double b_kj = B(k, j); 
                        for (int i = iRowBlkA; i < std::min(A.nbRows, iRowBlkA + szBlock); ++i) {
                            C(i, j) += A(i, k) * b_kj;
                        }
                    }
                }
            }
            
            int findOptimalBlockSize(const Matrix& A, const Matrix& B) {
                std::vector<int> sizes = {16, 32, 64, 128, 256, 512, 1024, 2048};
                double best_time = 1e9;
                int best_size = 64; 
            
                for (int sz : sizes) {
                    Matrix C(A.nbRows, B.nbCols, 0.0);
                    double start = omp_get_wtime();
                    
                    for (int I = 0; I < A.nbRows; I += sz) {
                        for (int J = 0; J < B.nbCols; J += sz) {
                            for (int K = 0; K < A.nbCols; K += sz) {
                                prodSubBlocks(I, J, K, sz, A, B, C);
                            }
                        }
                    }
                    
                    double duration = omp_get_wtime() - start;
                std::cout << "Durée correspondante à la  Taille de bloc " << sz << ": " << duration << std::endl;
                    if (duration < best_time) {
                        best_time = duration;
                        best_size = sz;
                    }
                }
                
                std::cout << "[OPTIM] Taille de bloc optimale: " << best_size << std::endl;
                return best_size;
            }
            } // namespace

            Matrix operator*(const Matrix& A, const Matrix& B) {
                Matrix C(A.nbRows, B.nbCols, 0.0);
                const int szBlock = findOptimalBlockSize(A, B);
            
                for (int I = 0; I < A.nbRows; I += szBlock) {       // Parcours lignes A
                    for (int J = 0; J < B.nbCols; J += szBlock) {   // Parcours colonnes B
                        for (int K = 0; K < A.nbCols; K += szBlock) { // Parcours commun
                            prodSubBlocks(I, J, K, szBlock, A, B, C);
                        }
                    }
                }
                return C;
            }