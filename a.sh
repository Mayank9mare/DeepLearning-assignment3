#!/bin/bash

cmd1=`rm -rf logs`
echo $cmd1

cmd2=`rm -rf images`
echo $cmd2

cmd3=`mkdir logs`
echo $cmd3

cmd4=`mkdir images`
echo $cmd4

cmd5=`nohup python3 dcgan.py > output.txt &`
echo $cmd5