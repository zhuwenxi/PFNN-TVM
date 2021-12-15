//
//  timer.h
//  tvm_runtime_packed
//
//  Created by haidonglan on 2021/7/20.
//

#ifndef timer_h
#define timer_h

#include <iostream>
#include <ctime>
#include <cmath>

class Timer
{
public:
    void start()
    {
        clock_gettime(CLOCK_MONOTONIC, &m_StartTime);
        m_bRunning = true;
    }
    
    void stop()
    {

        clock_gettime(CLOCK_MONOTONIC, &m_EndTime);
        m_bRunning = false;
    }
    
    double elapsedMilliseconds()
    {
        timespec endTime;
        
        if(m_bRunning)
        {
            clock_gettime(CLOCK_MONOTONIC, &endTime);
        }
        else
        {
            endTime = m_EndTime;
        }
        
//        return std::chrono::duration_cast<std::chrono::milliseconds>(endTime - m_StartTime).count();
        return (endTime.tv_sec - m_StartTime.tv_sec) * 1.0e3 + (endTime.tv_nsec - m_StartTime.tv_nsec) / 1.0e6;
    }
    
    double elapsedSeconds()
    {
        return elapsedMilliseconds() / 1000.0;
    }

private:
    timespec m_StartTime;
    timespec m_EndTime;
    bool     m_bRunning = false;
};


#endif /* timer_h */
