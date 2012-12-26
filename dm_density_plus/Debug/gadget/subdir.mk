################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../gadget/gadgetreader.cpp \
../gadget/gadgetwriter.cpp 

C_SRCS += \
../gadget/read_utils.c 

OBJS += \
./gadget/gadgetreader.o \
./gadget/gadgetwriter.o \
./gadget/read_utils.o 

C_DEPS += \
./gadget/read_utils.d 

CPP_DEPS += \
./gadget/gadgetreader.d \
./gadget/gadgetwriter.d 


# Each subdirectory must supply rules for building sources it contributes
gadget/%.o: ../gadget/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

gadget/%.o: ../gadget/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C Compiler'
	gcc -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


