---
layout: page
title: note
permalink: /note/
---  

1. 
```
z = x*y
判断z是否溢出，
!x || z/x==y
无符号:
高n位全为0，不溢出
带符号：
高n为全为0或者全为1，且与低n位最高位相同，不溢出
1111 1000
加减不分无符号，带符号
除法：
带符号:
加上偏移量(2*k-1),在右移k位，在截断  
```  
2. 更新pip  
python -m pip  install --upgrade pip  

3. static_cast强制转换 double temp;static_cast<int>temp;    
内联函数inline,简单调用  

<li>{{ site.date }}</li>
<li>{{ site.time }}</li>