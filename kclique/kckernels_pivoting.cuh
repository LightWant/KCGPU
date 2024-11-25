#pragma once

/* __launch_bounds__(BLOCK_DIM_X, 16):
BLOCK_DIM_X specifies the maximum number of threads per block that this kernel is expected to use.
16 (the optional second argument) specifies the minimum number of blocks per multiprocessor (SM) that the kernel is expected to run with.
*/

template <typename T, uint BLOCK_DIM_X, uint CPARTSIZE>
__launch_bounds__(BLOCK_DIM_X, 16) 
__global__ void
kckernel_node_block_warp_binary_pivot_count(
    uint64* counter,
    graph::COOCSRGraph_d<T> g,
    const  graph::GraphQueue_d<T, bool>  current,
    T* current_level,
    uint64* cpn,
    T* levelStats,
    T* adj_enc,

    T* possible,
    T* level_index_g,
    T* level_count_g,
    T* level_prev_g,
    T* level_r,
    T* level_d,
    T* level_tmp,
    unsigned long long* nCR
)
{
    //will be removed later
    constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
    const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
    const size_t lx = threadIdx.x % CPARTSIZE;

    //__shared__ T  level_offset[numPartitions], level_item_offset[numPartitions]; //for l and p
    __shared__ T level_pivot[512];//512 > max depth
    // __shared__ uint64 clique_count[numPartitions];
    // __shared__ uint64 path_more_explore;
    __shared__ bool path_more_explore;
    __shared__ T l;
    __shared__ uint64 maxIntersection;
    __shared__ uint32_t  sm_id, levelPtr;
    __shared__ T src, srcStart, srcLen;
    __shared__ bool  partition_set[numPartitions];

    __shared__ T num_divs_local, encode_offset, *encode;
    __shared__ T *pl, *cl;
    __shared__ T *level_count, *level_index, *level_prev_index, *rsize, *drop;

    __shared__ T lo, level_item_offset;

    __shared__ T maxCount[numPartitions], maxIndex[numPartitions], partMask[numPartitions];

    __shared__ 	T lastMask_i, lastMask_ii;

    //Only the first thread (threadIdx.x == 0) within each thread block executes the following code
    if (threadIdx.x == 0)
    {
        sm_id = __mysmid();
        T temp = 0;
        //The function atomicCAS returns the original value at levelStats[(sm_id * CBPSM) + temp]
        //if levelStats[(sm_id * CBPSM) + temp] == 0, levelStats[(sm_id * CBPSM) + temp]= 1
        
        //levelStats is num_SMs * conc_blocks_per_SM
        //(sm_id * CBPSM) + temp 指向第一个为0的block时停止
        //ensures that each block finds a unique temp index that it can claim exclusively.
        //This technique is useful in scenarios where each block needs a unique location within shared or global memory to store or retrieve its data safely without overwriting data from other blocks.
        while (atomicCAS(&(levelStats[(sm_id * CBPSM) + temp]), 0, 1) != 0)
        {
            temp++;
        }
        levelPtr = temp;
    }
    __syncthreads();

    //gridDim.x  is the total number of  blocks in the grid.
    //当前block负责第 blockIdx.x blockIdx.x+gridDim.x ... 点
    //每个block
    for (unsigned long long i = blockIdx.x; i < (unsigned long long)current.count[0]; i += gridDim.x)
    {
        __syncthreads();
        //block things
        if (threadIdx.x == 0)
        {
            src = current.queue[i];
            srcStart = g.rowPtr[src];
            srcLen = g.rowPtr[src + 1] - srcStart;

            //printf("src = %u, srcLen = %u\n", src, srcLen);
        
            num_divs_local = (srcLen + 32 - 1) / 32;
            // num_divs_local = (srcLen + CPARTSIZE - 1) / CPARTSIZE;
            /**
             * adj_enc is num_SMs * conc_blocks_per_SM * maxDegree.gdata()[0] * num_divs
             * uint num_divs = (maxDegree.gdata()[0] + 32 - 1) / 32;
             */
            encode_offset = sm_id * CBPSM * (MAXDEG * NUMDIVS) + levelPtr * (MAXDEG * NUMDIVS);
            encode = &adj_enc[encode_offset  /*srcStart[wx]*/];

            lo = sm_id * CBPSM * (/*numPartitions **/ NUMDIVS * MAXDEG) + levelPtr * (/*numPartitions **/ NUMDIVS * MAXDEG);
            cl = &current_level[lo/*level_offset[wx]/* + wx * (NUMDIVS * MAXDEG)*/];
            //num_SMs * conc_blocks_per_SM *  max_level * num_divs; max_level==maxDegree.gdata()[0];
            pl = &possible[lo/*level_offset[wx] /*+ wx * (NUMDIVS * MAXDEG)*/];

            level_item_offset = sm_id * CBPSM * (/*numPartitions **/ MAXDEG) + levelPtr * (/*numPartitions **/ MAXDEG);
            level_count = &level_count_g[level_item_offset /*+ wx*MAXDEG*/];
            level_index = &level_index_g[level_item_offset /*+ wx*MAXDEG*/];
            level_prev_index = &level_prev_g[level_item_offset /*+ wx*MAXDEG*/];
            rsize = &level_r[level_item_offset /*+ wx*MAXDEG*/]; // will be removed
            drop = &level_d[level_item_offset /*+ wx*MAXDEG*/];  //will be removed

            level_count[0] = 0;
            level_prev_index[0] = 0;
            level_index[0] = 0;
            l = 2;
            rsize[0] = 1;
            drop[0] = 0;

            level_pivot[0] = 0xFFFFFFFF;

            maxIntersection = 0;

            lastMask_i = srcLen / 32;
            lastMask_ii = (1 << (srcLen & 0x1F)) - 1;
        }
        __syncthreads();
        //Encode Clear
        //const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
        // const size_t lx = threadIdx.x % CPARTSIZE;
        //encode is MAXDEG * NUMDIVS;  NUMDIVS=(maxDegree.gdata()[0] + 32 - 1) / 32;
        // num_divs_local = (srcLen + 32 - 1) / 32;
        // constexpr T numPartitions = BLOCK_DIM_X / CPARTSIZE;
        //G[i][j] is encoded in the 32bit encode[j*num_divs_local + i/32], 1 << i%32
        for (T j = wx; j < srcLen; j += numPartitions)
        {
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                encode[j * num_divs_local + k] = 0x00;
            }
        }
        __syncthreads();
        //Full Encode
        for (T j = wx; j < srcLen; j += numPartitions)
        {
            // if(current.queue[i] == 40 && lx == 0)
            // 	printf("%llu -> %u, ", j, g.colInd[srcStart + j]);
            graph::warp_sorted_count_and_encode_full<WARPS_PER_BLOCK, T, true, CPARTSIZE>(&g.colInd[srcStart], srcLen,
                &g.colInd[g.rowPtr[g.colInd[srcStart + j]]], g.rowPtr[g.colInd[srcStart + j] + 1] - g.rowPtr[g.colInd[srcStart + j]],
                j, num_divs_local, encode);
        }
        __syncthreads(); //Done encoding

        //Find the first pivot
        if(lx == 0)
        {
            maxCount[wx] = 0;
            maxIndex[wx] = 0xFFFFFFFF;
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
        __syncthreads();

        // numPartitions =  BLOCK_DIM_X / CPARTSIZE;
        // wx = threadIdx.x / CPARTSIZE; // which warp in thread block
        // num_divs_local = (srcLen + 32 - 1) / 32;
        // lx = threadIdx.x % CPARTSIZE;

        /**
         * 每CPARTSIZE个线程看作一组，一共numPartitions组，srcLen分为srcLen/numPartitions块
         * 每块numPartitions组一起跑，
         */
        for (T j = wx; j < srcLen; j += numPartitions)
        {
            uint64 warpCount = 0;
            for (T k = lx; k < num_divs_local; k += CPARTSIZE)
            {
                /**
                 * When j+=numPartitions,j * num_divs_local变大 numPartitions*num_divs_local
                 * 
                 */
                warpCount += __popc(encode[j * num_divs_local + k]);
            }
            reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);

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
                atomicMin(&(level_pivot[0]),maxIndex[wx]);
            }
        }
        __syncthreads();

        //Prepare the Possible and Intersection Encode Lists
        uint64 warpCount = 0;
        //distributing work evenly across the block.
        //it counts the bits where encode has 0s and m has 1s.
        /**
         * lastMask_i = srcLen / 32;
            lastMask_ii = (1 << (srcLen & 0x1F)) - 1;
         */
        for (T j = threadIdx.x; j < num_divs_local && maxIntersection > 0; j += BLOCK_DIM_X)
        {
            T m = (j == lastMask_i) ? lastMask_ii : 0xFFFFFFFF;
            pl[j] = ~(encode[(level_pivot[0])*num_divs_local + j]) & m;
            cl[j] = 0xFFFFFFFF;
            warpCount += __popc(pl[j]);
        }

        /**
         * reduce_part is warp-level without requiring shared memory
         * only thread 0 (lx==0) in each warp holds the total sum of warpCount for that warp
         * level_count[0] is the count of non-neighbors of the pivot level_pivot[0]
         */
        reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);
        if(lx == 0 && threadIdx.x < num_divs_local)
        {
            atomicAdd(&(level_count[0]), (T)warpCount);
        }
        __syncthreads();

        //Explore the tree
        /**
         * level_count[0] = 0;
            level_prev_index[0] = 0;
            level_index[0] = 0;
            l = 2;
            rsize[0] = 1;
            drop[0] = 0;
         */
        while((level_count[l - 2] > level_index[l - 2]))
        {

            if(threadIdx.x == 0) //Temp level Counter, the count of level
                atomicAdd(&cpn[sm_id], 1);

            T maskBlock = level_prev_index[l- 2] / 32;
            /**
             * 低部分的都被设置为0
             */
            T maskIndex = ~((1 << (level_prev_index[l - 2] & 0x1F)) -1);

            /**
             * __ffs 返回第一个为1的点
             * newIndex指向第一个非邻居
             * int x = 0b101100; // Binary representation
             * int pos = __ffs(x); // pos would be 3
             * int x = 0b000000; // Binary representation
             * int pos = __ffs(x); // pos would be 0
             */
            T newIndex = __ffs(pl[num_divs_local*(l-2) + maskBlock] & maskIndex);

            maskIndex = 0xFFFFFFFF;
            while(newIndex == 0)
            {
                
                maskBlock++;
                newIndex = __ffs(pl[num_divs_local*(l-2) + maskBlock] & maskIndex);
            }
            /**
             * newIndex 是图上点的编号了
             */
            newIndex =  32*maskBlock + newIndex - 1;
            /**
             * ~((1 << (newIndex & 0x1F)) - 1)是比newIndex高的点
             * ~pl[num_divs_local*(l-2) + maskBlock] 是上一个
             */
            T sameBlockMask = (~((1 << (newIndex & 0x1F)) - 1))   | ~pl[num_divs_local*(l-2) + maskBlock];
            __syncthreads();
            if (threadIdx.x == 0)
            {
                level_prev_index[l - 2] = newIndex + 1;
                level_index[l - 2]++;
                level_pivot[l - 1] = 0xFFFFFFFF;
                path_more_explore = false;
                maxIntersection = 0;
                rsize[l-1] = rsize[l-2] + 1;
                drop[l-1] = drop[l-2];
                if(newIndex == level_pivot[l-2])
                    drop[l-1] = drop[l-2] + 1;
            }
            __syncthreads();
            //assert(level_prev_index[l - 2] == newIndex + 1);

            /**
             * drop是pivot的数量
             * rsize[l-1] - drop[l-1]是必选点的数量
             */
            if(rsize[l-1] - drop[l-1] > KCCOUNT)
            {	
                __syncthreads();
                //printf("Stop Here, %u %u\n", rsize[l-1], drop[l-1]);
                if(threadIdx.x == 0)
                {   
                    //printf, go back
                    while (l > 2 && level_index[l - 2] >= level_count[l - 2])
                    {
                        (l)--;
                    }
                }
                __syncthreads();
            }
            else
            {
                // Now prepare intersection list

                /**cl[j] = 0xFFFFFFFF; */
                T* from = &(cl[num_divs_local * (l - 2)]);
                T* to =  &(cl[num_divs_local * (l - 1)]);

                /**
                 * 从newIndex的邻居里面去掉之前枚举过的点
                 */
                for (T k = threadIdx.x; k < num_divs_local; k += BLOCK_DIM_X)
                {
                    to[k] = from[k] & encode[newIndex* num_divs_local + k];
                    //remove previous pivots from here
                    /**
                     * pl[j] = ~(encode[(level_pivot[0])*num_divs_local + j]) & m;
                     * if (maskBlock < k), remove the 1bits in pl[num_divs_local*(l-2) + k] from to[k]
                     * else if maskBlock > k, does not change to[k]
                     * else the sameBlock
                     * 
                     * 第一次搜索树上这里pl是pivot的非邻居，也就是要枚举的点，maskBlock < k的pivot的非邻居就是枚举过的点
                     * 后面搜索树这里 pl 是 pl[(l-1)*num_divs_local + j] = ~(encode[level_pivot[l - 1] * num_divs_local + j]) & to[j] & m;
                     * 也就是上一层的pivot的非邻居
                     */
                    to[k] = to[k] & ( (maskBlock < k) ? ~pl[num_divs_local*(l-2) + k] : ( (maskBlock > k) ? 0xFFFFFFFF:  sameBlockMask) );
                }
                if(lx == 0)
                {	
                    partition_set[wx] = false;
                    maxCount[wx] = srcLen + 1; //make it shared !!
                    maxIndex[wx] = 0;
                }
                __syncthreads();
                //////////////////////////////////////////////////////////////////////
                //Now new pivot generation, then check to extend to new level or not

                //T limit = (srcLen + numPartitions -1)/numPartitions;
                /**
                 * const int wx = threadIdx.x / CPARTSIZE; // which warp in thread block
                 * 每个wrap一起处理一个点，也就是wx相同的线程一起
                 */
                for (T j = wx; j < /*numPartitions*limit*/srcLen; j += numPartitions)
                {
                    uint64 warpCount = 0;
                    T bi = j / 32;
                    T ii = j & 0x1F;
                    if( (to[bi] & (1<<ii)) != 0)
                    {
                        for (T k = lx; k < num_divs_local; k += CPARTSIZE)
                        {
                            warpCount += __popc(to[k] & encode[j * num_divs_local + k]);
                        }
                        reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);
                        if(lx == 0 && maxCount[wx] == srcLen + 1)
                        {
                            partition_set[wx] = true;
                            path_more_explore = true; //shared, unsafe, but okay
                            maxCount[wx] = warpCount;
                            maxIndex[wx] = j;
                        }
                        else if(lx == 0 && maxCount[wx] < warpCount)
                        {
                            maxCount[wx] = warpCount;
                            maxIndex[wx] = j;
                        }	
                    }
                }
        
                __syncthreads();
                if(!path_more_explore)
                {
                    __syncthreads();
                    if(threadIdx.x == 0)
                    {	
                        if(rsize[l-1] >= KCCOUNT)
                        {
                            T c = rsize[l-1] - KCCOUNT;
                            unsigned long long ncr = nCR[ drop[l-1] * 401 + c  ];
                            atomicAdd(counter, ncr/*rsize[l-1]*/);
                        }
                        //printf, go back
                        while (l > 2 && level_index[l - 2] >= level_count[l - 2])
                        {
                            (l)--;
                        }
                    }
                    __syncthreads();
                }
                else
                {
                    if(lx == 0 && partition_set[wx])
                    {
                        atomicMax(&(maxIntersection), maxCount[wx]);
                    }
                    __syncthreads();

                    if(lx == 0 && maxIntersection == maxCount[wx])
                    {	
                            atomicMin(&(level_pivot[l-1]), maxIndex[wx]);
                    }
                    __syncthreads();

                    uint64 warpCount = 0;
                    for (T j = threadIdx.x; j < num_divs_local; j += BLOCK_DIM_X)
                    {
                        T m = (j == lastMask_i) ? lastMask_ii : 0xFFFFFFFF;
                        pl[(l-1)*num_divs_local + j] = ~(encode[level_pivot[l - 1] * num_divs_local + j]) & to[j] & m;
                        warpCount += __popc(pl[(l-1)*num_divs_local + j]);
                    }
                    reduce_part<T, CPARTSIZE>(partMask[wx], warpCount);
                    __syncthreads(); // Need this for degeneracy > 1024
                    
                    if(threadIdx.x == 0)
                    {
                        l++;
                        level_count[l-2] = 0;
                        level_prev_index[l-2] = 0;
                        level_index[l-2] = 0;
                    }

                    __syncthreads();
                    if(lx == 0 && threadIdx.x < num_divs_local)
                    {
                        atomicAdd(&(level_count[l-2]), warpCount);
                    }
                }
            }
            __syncthreads();
            /////////////////////////////////////////////////////////////////////////
        }
    }

    __syncthreads();
    if (threadIdx.x == 0)
    {
        atomicCAS(&levelStats[sm_id * CBPSM + levelPtr], 1, 0);
    }
}
