#!/bin/sh
APP_NAME='deep-framework-spring-boot'
PID=$(ps -ef|grep ${APP_NAME}|grep -v grep|awk '{print $2}')
if [ ${PID} ]; then
    echo Kill ${APP_NAME} Process!
    kill -9 $PID
fi
nohup java -Xms10g -Xmx20G -Djava.util.concurrent.ForkJoinPool.common.parallelism=24 -jar ${APP_NAME}.jar > ${APP_NAME}.out 2>&1 &