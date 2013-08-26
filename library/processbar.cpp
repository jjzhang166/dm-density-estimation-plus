#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "processbar.h"

using namespace std;

ProcessBar::ProcessBar(int maxvalue, int type){
	maxvalue_ = maxvalue;
	currentvalue_ = 0;
	currentpercent_ = 0;
	type_ = type;
}


void ProcessBar::start(){
	if(type_ == 0){
		fprintf(stderr,"=========[---10---20---30---40---50---60---70---80---90--100]========\n");
		fprintf(stderr,"=========[");
	}else if(type_ == 1){
		fprintf(stderr,"Computation starting ...\n");
	}

}

void ProcessBar::setvalue(int value){
	double rate = (double ) value / (double) maxvalue_;
	if(rate > currentvalue_){
		currentvalue_ =  rate;
		int percent = (int)ceil(rate * 50);
		if(percent > currentpercent_){
			int i;
			for(i = currentpercent_; i < percent; i++){
				if(type_ == 0){
					std::cout<<"<";
				}else if(type_ == 1){
					fprintf(stderr,"%i percent finished ...\n", (int)ceil(rate * 100));
				}

				std::cout.flush();
			}
			currentpercent_ = percent;
		}
	}
}

void ProcessBar::end(){
	if(type_ == 0){
		fprintf(stderr,"]========\n");
	}else if(type_ == 1){
		fprintf(stderr,"Computation Finishing ...\n");
	}
}
