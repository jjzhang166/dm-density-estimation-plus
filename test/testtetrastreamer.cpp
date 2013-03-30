#include "indtetrastream.h"

int main(){
    int inputmemgrid = 16;
    int parttype = 1;
    int datagridsize = 16;
    bool isHighMem = true;
    bool isAllData = false;
    bool isVelocity = true;
    bool isCorrection = true;
    bool isInOrder = false;
    TetraStreamer streamer("multires_150",
                           inputmemgrid,
                           parttype,
                           datagridsize,
                           isHighMem,
                           isAllData,
                           isVelocity,
                           isCorrection,
                           isInOrder);
    int count = 0;
    while(streamer.hasNext()){
        int nums;
        Tetrahedron * tetras;
        tetras = streamer.getNext(nums);
        /*for(int i= 0; i < nums; i++){
            printf("%d %f     \n", i, tetras[i].volume);
            printf("%f %f %f\n",tetras[i].v1.x, tetras[i].v1.y, tetras[i].v1.z);
            printf("%f %f %f\n",tetras[i].v2.x, tetras[i].v2.y, tetras[i].v2.z);
            printf("%f %f %f\n",tetras[i].v3.x, tetras[i].v3.y, tetras[i].v3.z);
            printf("%f %f %f\n",tetras[i].v4.x, tetras[i].v4.y, tetras[i].v4.z);
        
        }*/
        count += nums;
    }
    printf("%d\n", count);
    
    return 0;
}
