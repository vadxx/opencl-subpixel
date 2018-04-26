//
//  main.cpp
//  newPixelCL
//
//  Created by Anton Volkov on 19/04/2018.
//  Copyright © 2018 Anton Volkov. All rights reserved.
//
#define __CL_ENABLE_EXCEPTIONS
#define DATA_SIZE 1024
#include <fstream>
#include <iostream>
#include "cl.hpp"
#include <opencv2/highgui.hpp>
#include <vector>
using namespace std;

void InitKernel(vector <cl::Device> &devices, cv::Mat image, int rows, int cols)
{
    //estimate_new_position
    cv::Mat resp = image;
    std::vector<cv::Mat> ch;
    cv::split(resp, ch);
    
    cv::Mat ch0 = ch[0];
    cv::Point max_loc;
    cv::minMaxLoc(ch0, NULL, NULL, NULL, &max_loc);

    //Create Context and Queue
    cl::Context context(devices);
    cl::CommandQueue queue(context, devices[0]);
    
    //Load File
    string src = "mykernel.cl";
    
    //Create Program
    cl::Program program(context, src);
    program.build(devices);
    
    //Create Kernel
    cl::Kernel mykernel(program, "sub_pixel");
    
    //Init args
    float * matr = (float*)calloc(540*960, sizeof(float));
    int direction = 0;
    
    //Fill matrix
    for (int i = 0; i < rows - 1; i++)
    {
        for (int j = 0; j < cols - 1; j++)
            matr[i * cols + j] = image.at<float>(i,j);
    }
    
    cout << " -- OpenCL -- " << endl;
    //Allocate Memory

    cl::Buffer Matr(context, CL_MEM_READ_WRITE, 540*960 * sizeof(float));
    cl::Buffer Buff(context, CL_MEM_READ_WRITE, 3 * sizeof(float));
    
    queue.enqueueWriteBuffer(Matr, CL_TRUE, 0, 540 * 960 * sizeof(float), matr);

    //Setting args for Kernel
    int iArg = 0;
    mykernel.setArg(iArg++, Matr);
    mykernel.setArg(iArg++, Buff);
    mykernel.setArg(iArg++, 0); //  Direction: 0 - vertical; 1 - horizontal;
    mykernel.setArg(iArg++, max_loc.x); //  pixelX
    mykernel.setArg(iArg++, max_loc.y); //  pixelY
    mykernel.setArg(iArg++, rows - 1);
    mykernel.setArg(iArg++, cols - 1);

    //Set kernel in queue and Execute it
    queue.enqueueNDRangeKernel(mykernel, cl::NullRange, cl::NDRange(10, 10));
    //    cl::NDRange(540, 960)
    
    queue.finish();
    float *buff = (float*)calloc(3, sizeof(float));
    // Read buffer C into a local list
    queue.enqueueReadBuffer(Matr, CL_TRUE, 0, 540*960 * sizeof(float), matr);
    queue.enqueueReadBuffer(Buff, CL_TRUE, 0, 3 * sizeof(float), buff);
    
    cout << " -- Out: -- " << endl;
    for (int i = 0; i < 10; i++) {
        cout << " Matr[" << i << "]:\t" << matr[i] << endl;
    }
    cout<<"p.x: "<<max_loc.x<<endl;
    cout<<"p.y: "<<max_loc.y<<endl;
    
    cout << "Direction:\t" << direction << endl;
    cout << "Rows:\t\t" << rows << endl;
    cout << "Cols:\t\t" << cols << endl;
    cout << " - Buff: - " << endl;
    cout << "Delta:\t" << buff[0] << endl;
    cout << "PixelX:\t\t" << buff[1] << endl;
    cout << "PixelY:\t\t" << buff[2] << endl;
}

int main()
{
    int matRows = 0;
    int matCols = 0;
    cv::Mat image;
    
    //Read Image
    try {
        image = cv::imread("rr.jpg");
        matRows = image.rows;
        matCols = image.cols;
        cout << "Image readed" << endl;
    }
    catch (cl::Error error) {
        cout << "Image not readed:" << error.what() << "(" << error.err() << ")" << endl;
    }
    
    //Get available platform and device
    vector <cl::Platform> platforms;
    vector <cl::Device> devices;
    
    cl::Platform::get(&platforms);
    for (const auto &p : platforms) {
        cout << "Platform:\t" << p.getInfo<CL_PLATFORM_NAME>() << endl;
        p.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        for(const auto &d : devices)
            cout << d.getInfo<CL_DEVICE_NAME>() << endl;
    }
    
    try {
        InitKernel(devices, image, matRows, matCols);
    } catch (cl::Error error) {
        cout << "Error:\t" << error.what() << "( " << error.err() << " )" << endl;
    }
    return 0;
}
