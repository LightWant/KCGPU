#pragma once

#include "common.cuh"
#include "../include/TriCountPrim.cuh"

#include "../include/CSRCOO.cuh"

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE, uint MAXDEPTH, uint SL/*Start Level*/>
__launch_bounds__(BLOCK_DIM_X, 16)
__global__ void
kckernel_node_block_warp_binary_count_lc_v5(
    uint64* counter,
    graph::COOCSRGraph_d<T> g,
    const  graph::GraphQueue_d<T, bool>  current,
    T* current_level, T* current_level_largeC,
    uint64* cpn,
    T* levelStats,
    T* adj_enc,
    // uint64* globalCounter,
    T * largeCs,
    T * toDoLargeCs,
    unsigned long long* nCR)
{
    //Variables
    constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
    const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
    const size_t lx = threadIdx.x % CPARTSIZE;
    
    __shared__ unsigned short level_count[numPartitions][MAXDEPTH];
    __shared__ unsigned short level_prev_index[numPartitions][MAXDEPTH];

    // __shared__ T  level_offset[numPartitions];
    __shared__ T  level_offset;
    __shared__ uint64 clique_count[numPartitions];
    __shared__ T l[numPartitions];
    __shared__ uint32_t  sm_id, levelPtr;
    __shared__ T src, srcStart, srcLen;

    __shared__ T maxCount[numPartitions], maxIndex[numPartitions], partMask[numPartitions];

    __shared__ T num_divs_local, encode_offset, *encode;

    // __shared__ T largeCs[gridDim.x * NUMDIVS];
    __shared__ T largeCOffset, maxDegNode, toDoNodesCount, largeCliqueSize, maxIntersection, *largeC, *toDoLargeC;

    #define LA l[wx] - SL

    reserve_space<T>(sm_id, levelPtr, levelStats);
    __syncthreads();


     if(threadIdx.x == 0) 
        level_offset = sm_id * CBPSM * (numPartitions * NUMDIVS * MAXDEPTH) + levelPtr * (numPartitions * NUMDIVS * MAXDEPTH);
    __syncthreads();

    if(lx == 0)
    {
        partMask[wx] = CPARTSIZE ==32? 0xFFFFFFFF : (1 << CPARTSIZE) - 1;
        partMask[wx] = partMask[wx] << ((wx%(32/CPARTSIZE)) * CPARTSIZE);
        /**
         * A case CPARTSIZE==8
         * wx == 0: 00000000 00000000 00000000 11111111
         * wx == 1: 00000000 00000000 11111111 00000000
         * ...
         * wx == 127/8 == 15, wx%(32/CPARTSIZE)==3: 11111111 00000000 00000000 00000000
         */
    }
    
    __shared__ unsigned long long cTri2; if(threadIdx.x == 0)cTri2 = 0;
    __syncthreads();

// #define DEBUG
#ifdef DEBUG
T xxx = 84;
__shared__ T cTri; if(threadIdx.x == 0)cTri = 0;

__syncthreads();
#endif
    for (unsigned long long i = blockIdx.x; i < (unsigned long long)current.count[0]; i += gridDim.x)
    {
        //Read Node Info
        if (threadIdx.x == 0)
        {
            src = current.queue[i];
            srcStart = g.rowPtr[src];
            srcLen = g.rowPtr[src + 1] - srcStart;
            num_divs_local = (srcLen + 32 - 1) / 32;
            encode_offset = sm_id * CBPSM * (MAXDEG * NUMDIVS) + levelPtr * (MAXDEG * NUMDIVS);
            encode = &adj_enc[encode_offset];
            
            largeCOffset = sm_id * CBPSM * NUMDIVS + levelPtr * NUMDIVS;
            largeC = &largeCs[largeCOffset];
            toDoLargeC = &toDoLargeCs[largeCOffset];
        }
        __syncthreads();

        //Mask for each thread group in the warp
        // partMask[wx] = get_thread_group_mask<T, CPARTSIZE>(wx);
        

       
        T* cl = &current_level[level_offset + wx * (NUMDIVS * MAXDEPTH)];
        T* cl_largeC = &current_level_largeC[level_offset + wx * (NUMDIVS * MAXDEPTH)];
        
        //Build Induced Subgraph
        for (T j = wx; j < srcLen; j += numPartitions)
        {
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                encode[j * num_divs_local + k] = 0x00;
            }
        }
        __syncthreads();
        for (T j = wx; j < srcLen; j += numPartitions)
        {
            // if(current.queue[i] == 40 && lx == 0)
            // 	printf("%llu -> %u, ", j, g.colInd[srcStart + j]);
            graph::warp_sorted_count_and_encode_full<WARPS_PER_BLOCK, T, true, CPARTSIZE>(&g.colInd[srcStart], srcLen,
                &g.colInd[g.rowPtr[g.colInd[srcStart + j]]], g.rowPtr[g.colInd[srcStart + j] + 1] - g.rowPtr[g.colInd[srcStart + j]],
                j, num_divs_local, encode);
        }

        // build_induced_subgraph_noOri<T, CPARTSIZE>(wx, lx, g, srcStart, srcLen, numPartitions, num_divs_local, partMask[wx], encode);
        __syncthreads(); //Done encoding

        for(T j = threadIdx.x; j < num_divs_local; j += BLOCK_DIM_X)
        {
            largeC[j] = 0x0000;
        }

        // lastMask_i = srcLen / 32;
        // lastMask_ii = (1 << (srcLen & 0x1F)) - 1;
        for(T j = threadIdx.x; j < num_divs_local; j += BLOCK_DIM_X)
        {
            toDoLargeC[j] = 0xFFFFFFFF;
        }
        __syncthreads();
        if(threadIdx.x == 0) {
            if(srcLen & 0x1F) toDoLargeC[num_divs_local-1] = (1 << (srcLen & 0x1F)) - 1;
            else toDoLargeC[num_divs_local-1] = 0;
        }
        __syncthreads();
        
        toDoNodesCount = srcLen;


#ifdef DEBUG
if(threadIdx.x == 0) {
    T c = 0;
    for(T u = 0; u < srcLen; u++) {
        // T cc = 0;
        for(T v = u+1; v < srcLen; v++) {
            if(encode[u*num_divs_local + v/32]&(1<<(v&0x1F))) {
                for(T w = v+1; w < srcLen; w++) {
                    if(encode[u*num_divs_local + w/32]&(1<<(w&0x1F))) {
                        if(encode[v*num_divs_local + w/32]&(1<<(w&0x1F))) {
                            c++;
                        }
                    }
                }
                // cc++;
            }
        }
        // printf("nodeDeg:%u:%u\n", u, cc);
    }
    atomicAdd(&cTri, c);
    // printf("E253:%u b%u %p %u, i%u\n", 
    // encode[25*num_divs_local], blockIdx.x, (void*)encode, levelPtr, i);
}
    if(i == xxx && threadIdx.x == 0) {

    // printf("E251:%u b%u %p %u\nGraph %u:\n", 
    // encode[25*num_divs_local], blockIdx.x, (void*)encode, levelPtr, i);
        // printf("Graph %u:\n", i);
        for(T a = 0; a < srcLen; a++) {

            printf("%u:", a);
            for(T b = 0; b < srcLen; b++) {
                if(encode[a*num_divs_local + b/32] & (1<<(b&0x1F))) {
                    printf("%u ", b);
                    // encode[b*num_divs_local + a/32] |=  (1<<(a&0x1F));
    // assert(encode[b*num_divs_local + a/32] & (1<<(a&0x1F)));
                }
            }
            printf("\n");
// printf("E251:%u %u\n", encode[25*num_divs_local], blockIdx.x);
        }
    // printf("E252:%u %u %p\n", encode[25*num_divs_local], blockIdx.x, (void*)encode);
    }
    

    __syncthreads();
    
    T ite = 0;


#endif
        if(threadIdx.x == 0) largeCliqueSize = 0;

        do {
#ifdef DEBUG
    ite++;
#endif
    // __syncthreads();
            // if(threadIdx.x == 0 && blockIdx.x == 0)
            // printf("tid %u, bid %u, i %llu, ite %u\n", 
            //     threadIdx.x, blockIdx.x, i, ite++);
            /**
             * encode是邻接矩阵
             * largeC是已经找到的clique，初始化为0x0000
             * toDoLargeC是候选点，是largeC的点的公共邻居
             */
            // find_MaxDeg_Node<T, CPARTSIZE>(encode, largeC, 
            //     toDoLargeC, srcLen, num_divs_local);
            if(lx == 0)
            {
                maxCount[wx] = 0;
                if(threadIdx.x == 0) maxIntersection = 0, maxDegNode=0xFFFFFFFF;

                if(wx >= NUMDIVS) {
                    maxIndex[wx] = 0xFFFFFFFF;
                }
                else {
                    maxIndex[wx] = 0xFFFFFFFF;
                    T j = wx;
                    T newIndex = __ffs(toDoLargeC[j]);
                    while(j < NUMDIVS && newIndex == 0)
                    {
                        j += numPartitions;
                        newIndex = __ffs(toDoLargeC[j]);
                    }
                    maxIndex[wx] = j*32 + newIndex-1;
                    // __syncwarp(partMask[wx]);
                    // if(threadIdx.x == 0) maxDegNode = j;
#ifdef DEBUG
    if(i == xxx) {
        printf("wx %u, ini %u, ite %u, j %u, newIndex %u\n", 
        wx, maxIndex[wx], ite, j, newIndex);
    }
#endif            
                }

                
            }
            
            __syncthreads();

            for (T j = wx; j < srcLen; j += numPartitions)
            {
                T bkI = j / 32;
                T inBkI = j & 0x1F;
                if(!(toDoLargeC[bkI] & (1<<inBkI))) {
                    continue;
                }
                uint64 warpCount = 0;
                for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                {
        // if(i == 40) {
        //     printf("encode:%x,toDo:%x,warpCnt:%u\n",encode[j * num_divs_local + k],
        //     toDoLargeC[k], __popc(encode[j * num_divs_local + k] & toDoLargeC[k]));
        // }
                    warpCount += __popc(encode[j * num_divs_local + k] & toDoLargeC[k]);
                    // warpCount += __popc(encode[j * num_divs_local + k] & toDoLargeC[k]);
                }
                reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);

        // if(i == 40 && lx == 0) {
        //     printf("j:%u, deg %llu, computed by wrap %u, ite %u\n", 
        //         j, warpCount, wx, ite);
        // }

                if(lx == 0 && maxCount[wx] < warpCount)
                {
                    maxCount[wx] = warpCount;
                    maxIndex[wx] = j;
                }	
            }
            
            if(lx == 0)
            {
                atomicMax(&(maxIntersection), maxCount[wx]);
            }
            __syncthreads();
            if(lx == 0)
            {
                if(maxIntersection == maxCount[wx]) // unsafe, but okay I need any one with this max count
                {
                    atomicMin(&(maxDegNode),maxIndex[wx]);
                }
            }
            __syncthreads();
            if(threadIdx.x == 0) {
                largeCliqueSize++;
                largeC[maxDegNode/32] |= (1<<(maxDegNode&0x1F));
            }
                    

    // if(i == xxx && threadIdx.x == 0) {
        
    //     printf("here maxDegNode %u\n", maxDegNode);
    //     assert(!encode[maxDegNode* num_divs_local + maxDegNode/32] & (1<<(maxDegNode&0x1F)));
    //     // for (T k = 0; k < num_divs_local; k += 1) {
    //     //     if(encode[maxDegNode* num_divs_local + k/32] & (1<<(k&0x1F))) {
    //     //         printf("%u ", k);
    //     //     }printf("\n");
    //     // }
    // }

            for (T k = threadIdx.x; k < num_divs_local; k += BLOCK_DIM_X) {
    // if(i == xxx) {
    // printf("k %u, %u\n", k, encode[maxDegNode* num_divs_local + k]);
    // }
                toDoLargeC[k] = toDoLargeC[k] & encode[maxDegNode* num_divs_local + k];
            }
            
            if(threadIdx.x == 0) toDoNodesCount = 0;
            __syncthreads();

            // for (T k = threadIdx.x; k < num_divs_local; k += BLOCK_DIM_X) {
            //     if(toDoLargeC[k] > 0) {
            //         toDoNodesCount = 1;
            //         break;
            //     }
            // }

            uint64 warpCount222 = 0;
            for(T j = threadIdx.x; j < num_divs_local; j += BLOCK_DIM_X)
            {
                warpCount222 += __popc(toDoLargeC[j]);
            }
            reduce_part<T, CPARTSIZE>(partMask[wx], warpCount222);
            __syncthreads();
            // if(threadIdx.x < num_divs_local)
            //     atomicAdd(&toDoNodesCount, warpCount222);

            
            if(lx == 0 && threadIdx.x < num_divs_local)
            {
                atomicAdd(&toDoNodesCount, warpCount222);
            }
#ifdef DEBUG
    if(i == xxx && threadIdx.x == 0) {
        printf("toDo:");
        T c = 0;
        for(T a = 0; a < srcLen; a++) {
            if(toDoLargeC[a/32] & (1<<(a&0x1F)))
                printf("%u ", a), c++;
        }
        printf("\nc %u\n", c);
    }
#endif
            // if(threadIdx.x == 0) {
                // toDoNodesCount = warpCount222;
    // if(i == 40)
#ifdef DEBUG
    if(i == xxx && threadIdx.x == 0)
    printf("warpCount222 %u, ite %u\n", toDoNodesCount, ite);
#endif    
            // }
            __syncthreads();
            // if(toDoNodesCount == 1) break;

        } while(toDoNodesCount > 0);/**toDoLargeC is not empty */
        // if(threadIdx.x == 0 && blockIdx.x == 0)
        //     printf("%llu %u %llu\n", i, blockIdx.x, (unsigned long long)current.count[0]);
        
        if(threadIdx.x == 0 && largeCliqueSize >= KCCOUNT-1) {

            unsigned long long ncr = nCR[ largeCliqueSize * 401 + KCCOUNT-1  ];
#ifdef DEBUG
if(i == xxx) {
    printf("largeCliqueSize:%u\n", largeCliqueSize);
    printf("lC:");
    for(T a = 0; a < srcLen; a++) {
        if(largeC[a/32] & (1<<(a&0x1F)))
            printf("%u ", a);
    }
    printf("\nadd %llu\n", ncr);
}
#endif            
            atomicAdd(&cTri2, ncr/*rsize[l-1]*/);

            // atomicAdd(counter, ncr/*rsize[l-1]*/);

        }
        // continue; ;
        __syncthreads();

        /**
         * encode，除了largeClique，其余的点的encode改成有向图
         */

        for (T j = wx; j < srcLen; j += numPartitions)
        {
            T z = j/32;
            T zi = j&0x1F;
            if(largeC[z]&(1<<(zi))) continue;
            for (T k = lx; k < z; k += CPARTSIZE)
            {
                encode[j * num_divs_local + k] &= largeC[k];
            }
            if(lx == 0) {
                T mk = ~((1<<zi)-1);
                encode[j * num_divs_local + z] &= (largeC[z]|mk);
            }
        }
#ifdef DEBUG
    __syncthreads();
    if(i == xxx && threadIdx.x == 0) {
        printf("GraphModified %u:\n", i);
        for(T a = 0; a < srcLen; a++) {
            printf("%u:", a);
            for(T b = 0; b < srcLen; b++) {
                if(encode[a*num_divs_local + b/32] & (1<<(b&0x1F))) {
                    printf("%u ", b);
                }
            }
            printf("\n");
        }
    }
    __syncthreads();

#endif

// #define DDEBUG
        //Explore each subtree at the second level
        for (T j = wx; j < srcLen; j += numPartitions)
        {
            //Init Stack per thread group
            
            if(largeC[j/32]&(1<<(j&0x1F))) {
#ifdef DDEBUG
if(i == xxx && lx== 0) {
    printf("subtree %u pruned\n", j);
    // printf("E5:%u %u\n", encode[5*num_divs_local], encode[5*num_divs_local + 0/32] & (1<<(0&0x1F)));
}
#endif
                continue;
            }
            if (lx == 0)
            {
                level_count[wx][0] = 0;
                l[wx] = SL;
                level_count[wx][0] = 0;
                level_prev_index[wx][0] = 0;
                
                clique_count[wx] = 0;
                maxCount[wx] = 0;
            }
#ifdef DDEBUG
if(i == xxx && lx== 0) {
    printf("subtree %u, wx %u, counter %llu, %u %u\n", j, wx, *counter, srcLen, numPartitions);
}
#endif
            //Get number of elements for the next level (3rd)
            uint64 warpCount = 0;
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {

                warpCount += __popc(encode[j * num_divs_local + k] &(~largeC[k]));
#ifdef DDEBUG
if(i == xxx) {
    printf("j %u, wx %u-%u,  e %x, ~l %x, cap %x, pop %u, warpCount %llu\n", 
        j, wx, k, encode[j * num_divs_local + k], (~largeC[k]), encode[j * num_divs_local + k] &(~largeC[k])
        , __popc(encode[j * num_divs_local + k] &(~largeC[k])), warpCount);
}
#endif
            }
            reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);

            uint64 warpCount2 = 0;
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                cl_largeC[num_divs_local * (LA)+k] = encode[j * num_divs_local + k] &(largeC[k]);
                cl[num_divs_local * (LA)+k] = encode[j * num_divs_local + k]&(~largeC[k]);
                warpCount2 += __popc(encode[j * num_divs_local + k] &(largeC[k]));
            }
            reduce_part<T, CPARTSIZE>(partMask[wx], warpCount2);

            if (lx == 0) {
                atomicAdd(&maxCount[wx], warpCount2);
            }
            __syncwarp(partMask[wx]);

            
            if (lx == 0)
            {
                //For each subtree, check if we reached the level we are looking for
                if (l[wx] == KCCOUNT)
                {
#ifdef DEBUG
if(i == xxx) {
    printf("j %u, l[wx] == KCCOUNT, warpCount %llu\n", j, warpCount);
}
#endif
                    clique_count[wx] += warpCount;
                }
                else if (KCCOUNT > SL &&  warpCount > 0 &&
                    (warpCount+maxCount[wx] >= KCCOUNT - SL + 1 ))
                {
#ifdef DEBUG
if(i == xxx) {
    printf("j %u, level_count[wx][0] = warpCount %llu\n", j, warpCount);
}
#endif
                    level_count[wx][LA] = warpCount;
                    // for(T k = 0; k < num_divs_local; k++)
                    // cl[num_divs_local * (LA)+k] = encode[num_divs_local * (LA)+k];
                }
                
            }
            __syncwarp(partMask[wx]);
            if (lx == 0 && KCCOUNT+1>=l[wx] && maxCount[wx] >= KCCOUNT+1-l[wx]) {
#ifdef DDEBUG
if(i == xxx) {
    printf("j %u, nCR[%u*401 + %u] = %llu\n", 
        j, maxCount[wx], KCCOUNT+1-l[wx],
        nCR[maxCount[wx]*401 + KCCOUNT+1-l[wx]]);
}
#endif

                clique_count[wx] += nCR[maxCount[wx]*401 + KCCOUNT+1-l[wx]];
                //atomicAdd(counter, nCR[maxCount[wx]*401 + KCCOUNT+1-l[wx]]);
            }
            __syncwarp(partMask[wx]);

            // if(lx == 0 && level_count[wx][LA] > 0) {
            //     cl_largeC[num_divs_local * (LA)]
            // }
// T ite = 0;
            while (level_count[wx][LA] > 0)
            {
// ite++;
                // if(lx == 0) {
                //     atomicAdd(&cpn[sm_id], 1);
                // }
// if(ite > 0 && lx == 0)
// printf("%u-%u-%u-%u-%u, ", LA, level_count[wx][LA], sm_id, levelPtr, ite) ; 
                //Current and Next Level Lists
                // T* from = l[wx] == SL ? &(encode[num_divs_local * j]) : &(cl[num_divs_local * (LA)]);
                T* from = &(cl[num_divs_local * (LA)]);
                T* to = &(cl[num_divs_local * (LA + 1)]);
                T* from_largeC = &(cl_largeC[num_divs_local * (LA)]);
                T* to_largeC = &(cl_largeC[num_divs_local * (LA + 1)]);

                T maskBlock = level_prev_index[wx][LA] / 32;
                T maskIndex = ~((1 << (level_prev_index[wx][LA] & 0x1F)) -1);

                // T newIndex = l[wx] == SL ? get_next_sibling_index_maxked<T>(from, largeC, maskIndex, maskBlock) : get_next_sibling_index<T>(from, maskIndex, maskBlock);
                T newIndex = get_next_sibling_index<T>(from, maskIndex, maskBlock);
#ifdef DDEBUG
if(i == xxx && lx == 0) {
    printf("j %u, wx %u, preIdx %u, newIndex %u, LA %u\n", 
        j, wx, level_prev_index[wx][LA], newIndex, LA);
}
#endif
                if (lx == 0)
                {
                    level_prev_index[wx][LA] = newIndex + 1;
                    level_count[wx][LA]--;

                    maxCount[wx] = 0;
                }
                __syncwarp(partMask[wx]);

                //Intersect
                uint64 warpCount = 0;
                for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                {
                    // to[k] = from[k] & encode[newIndex* num_divs_local + k] & (~largeC[k]);
                    to[k] = from[k] & encode[newIndex* num_divs_local + k];
                    warpCount += __popc(to[k]);
                }
                reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);

// break;
                uint64 warpCount2 = 0;
                for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                {
                    to_largeC[k] = encode[newIndex * num_divs_local + k] &(from_largeC[k]);
                    warpCount2 += __popc(encode[newIndex * num_divs_local + k] &(from_largeC[k]));
                }
                reduce_part<T, CPARTSIZE>(partMask[wx], warpCount2);
                __syncwarp(partMask[wx]);
// continue;
// break;

                if (lx == 0)
                {
                    atomicAdd(&maxCount[wx], warpCount2);
                }
                __syncwarp(partMask[wx]);
                if (lx == 0 && KCCOUNT >= l[wx] && maxCount[wx] >= KCCOUNT-l[wx]) {

                    clique_count[wx] += nCR[maxCount[wx]*401 + KCCOUNT-l[wx]];
#ifdef DDEBUG
if(i == xxx) {
    printf("j %u, nCR[%u*401 + %u] = %llu, LA %u\n", 
        j, maxCount[wx], KCCOUNT-l[wx],
        nCR[maxCount[wx]*401 + KCCOUNT-l[wx]], LA);
}
#endif
                }
                

                //Decide Next Step: Count, Go Deeper, Go back
                if (lx == 0)
                {

                    if (l[wx] + 1 == KCCOUNT) {
                        clique_count[wx] += warpCount;
#ifdef DDEBUG
if(i == xxx) {
    printf("j %u, l[wx] == KCCOUNT, warpCount %llu\n", j, warpCount);
}
#endif
                    } 
                    else if (l[wx] + 1 < KCCOUNT && warpCount > 0 &&
                        (warpCount + maxCount[wx] >= KCCOUNT - l[wx]))
                    {
                        (l[wx])++;
                        level_count[wx][LA] = warpCount;
                        level_prev_index[wx][LA] = 0;
                    }
    
                    //Go back, if needed
                    while (l[wx] > SL && level_count[wx][LA] == 0 )
                    {
                        l[wx]--;
                    }
                }
                __syncwarp(partMask[wx]);
                
            }
            
            if (lx == 0)
            {
                atomicAdd(&cTri2, clique_count[wx]);
                
                // atomicAdd(&counter, clique_count[wx]);
#ifdef DDEBUG

if(i == xxx) {
    printf("counter+=%llu, wx %d, j %u, counter=%llu\n", 
        clique_count[wx], wx, j, *counter);
}
#endif
            }

            __syncwarp(partMask[wx]);
        }

// if (threadIdx.x == 0) printf("Graph %u:\n", i);
    }
 
    __syncthreads();
    if(threadIdx.x == 0) {
// printf("cTri2:%u,%u\n",  cTri2, blockIdx.x);
        atomicAdd(counter, cTri2);
    }
#ifdef DEBUG
if(threadIdx.x == 0)
printf("cTri:%u, %u,%u\n", cTri, cTri2, blockIdx.x);
#endif
    release_space<T>(sm_id, levelPtr, levelStats);
    // if(threadIdx.x == 0 && blockIdx.x > 621000) {
    //     // printf("block %u stop, grid %u, sm %u\n", 
    //     //     blockIdx.x, gridDim.x, sm_id);
    //     printf("block %u stop, sm %u\n", blockIdx.x, sm_id);
    // }
}
