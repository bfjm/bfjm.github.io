---
layout: post
title: ubuntu安装wps
date: 2018-05-13 16:00 +0800
categories: ubuntu
tags: install
---

Ubuntu18.04下面安装wps
-
<!--more-->
> sudo dpkg -i  wps-office_10.1.0.5672_a21_amd64.deb
提示需要依赖文件libpng12-0  
手动下载地址:  

http://archive.debian.org/debian/pool/main/libp/libpng/libpng12-0_1.2.27-2+lenny5_amd64.deb 

输入命令:
> sudo dpkg -i libpng12-0_1.2.27-2+lenny5_amd64.deb  

再次安装wps: 
> sudo dpkg -i  wps-office_10.1.0.5672_a21_amd64.deb  
ok到此完成