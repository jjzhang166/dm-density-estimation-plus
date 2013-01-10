#include <cmath>
#include "processbar.h"

using namespace std;

ProcessBar::ProcessBar(int maxvalue){
	maxvalue_ = maxvalue;
	currentvalue_ = 0;
	currentpercent_ = 0;
}


void ProcessBar::start(){
	printf("=========[---10---20---30---40---50---60---70---80---90--100]========\n");
	printf("=========[");
}

void ProcessBar::setvalue(int value){
	double rate = (double ) value / (double) maxvalue_;
	if(rate > currentvalue_){
		currentvalue_ =  rate;
		int percent = (int)ceil(rate * 50);
		if(percent > currentpercent_){
			int i;
			for(i = currentpercent_; i < percent; i++){
				std::cout<<"<";
				std::cout.flush();
			}
			currentpercent_ = percent;
		}
	}
}

void ProcessBar::end(){
	printf("]========\n");
}