---
layout: post
title: "蓝桥杯第十五届 C++ A组省赛部分题解"
date:   2024-12-15
tags: [容斥, 枚举]
comments: true
author: Shanggyx
---

### [真题链接](https://www.lanqiao.cn/paper/)

篮球杯官网现在支持 `C++17`，正式赛不知道是不是还是 `C++11`。

<!-- more -->

#### [因数计数](https://www.lanqiao.cn/problems/19706/learning/) 

这是比赛里的第四个编程题。

#### 题目大意

给定一个长度为 $n$ 的正整数数组 $a$，求有多少个四元组 $(i, j, k, l)$ 满足 $(a_i, a_j) \mid (a_k, a_l)$ 且 $i, j, k, l$ 互不相同。

- 其中 $(a_i, a_j) \mid (a_k, a_l)$ 表示 $a_i \mid a_k$ 且 $a_j \mid a_l$;
- $x \mid y$ 表示 $x$ 整除 $y$，例如 $2 \mid 4$.

#### 数据范围

- $1 \leq n, a_i \leq 10^5$.

#### Solution

这题很明显是一道容斥。在满足 $(a_i, a_j) \mid (a_k, a_l)$ 的基础上，我们令 $ABCDE$ 分别表示以下几个四元组。

- $A$ 表示 $i \neq k$ 且 $j \neq l$;
- $B$ 表示 $i = j$;
- $C$ 表示 $i = l$;
- $D$ 表示 $j = k$;
- $E$ 表示 $k = l$.

那么题目要求的答案就是 

$$
ans = A\ \overline{B}\ \overline{C}\ \overline{D}\ \overline{E} = A - \overline{B \cup C \cup D \cup E}.
$$

我们把后面的一堆 $\cup$ 展开，就会得到

$$
\begin{align*}
ans = A &- (AB + AC + AD + AE) \\
&+ (ABC + ABD + ABE + ACD + ACE + ADE) \\
&- (ABCD + ABCE + ABDE + ACDE) \\
&+ ABCDE.
\end{align*}
$$

这个式子看着吓人，其实一堆都是 $0$，或者有些是可以合并计算的。

由于 $\cap$ 得越多，集合越小，答案就越小，所以我们从下往上看。

##### 首先是 $ABCDE$。

将 $BCDE$ 代入上面的定义可以得到一个 $(i, i, i, i)$ 四元组，与 $A$ 的要求矛盾，因此这种情况结果是 $0$。

##### 其次是 $ABCD$ 这一行。

以 $ABCD$ 为例，将 $BCD$ 代入上面的定义，我们会得到 $(i,i,i,i)$，这与 $A$ 的要求矛盾。其它三个也都会产生这样的矛盾，因此这四种情况都是 $0$。

##### 然后是 $ABC$ 这一行。

我们把 $ABC,ABD,ACE,ADE$ 得到的四元组写到一起

$$
\begin{align*}
(i, i&, k, i) \\
(i, i&, i, l) \\
(i, j&, i, i) \\
(i, j&, j, j).
\end{align*}
$$

不难发现它们均违反了 $A$ 的要求，所以这四种都是 $0$。

对于 $ABE$ 来说，它产生的四元组是 $(i,i,k,k)$，这就相当于求有多少个二元组 $(i,k)$ 满足 $i \neq k$ 且 $a_i \mid a_k$。

我们可以先用 $O(n)$ 的时间预处理出每个 $x$ 出现了多少次，记为 $cnt[x]$；再用调和级数枚举的方法在 $O(n\log n)$ 的时间内预处理 $x$ 的倍数个数 $-1$（减一是因为要求 $i \neq k$），记为 $mul[x]$。

这样我们就可以得到这种情况的答案 

$$
\sum\limits cnt[x] \times mul[x].
$$

对于 $ACD$ 来说，它产生的四元组是 $(i,j,j,i)$，而要求是 $a_i \mid a_j$ 且 $a_j \mid a_i$，因此一定有 $a_i = a_j$。所以这种情况的答案就是 

$$
\sum\limits cnt[x] \times (cnt[x] - 1).
$$

##### 接着是 $AB$ 这一行。

考虑 $AB$ 对应的四元组 $(i,i,k,l)$，要求是 $a_i \mid a_k$ 且 $a_i \mid a_l$。我们上面预处理了倍数数量 $-1$，为 $mul[x]$，因此这种情况的答案为 

$$
\sum\limits cnt[x] \times mul[x] \times mul[x].
$$

考虑 $AE$ 对应的四元组 $(i,j,k,k)$，要求是 $a_i \mid a_k$ 且 $a_j \mid a_k$。我们模仿处理 $mul[x]$ 的过程，同样用 $O(n\log n)$ 的时间预处理出每个数 $x$ 的因数个数 $-1$，记为 $fac[x]$。于是这种情况的答案为

$$
\sum\limits cnt[x] \times fac[x] \times fac[x].
$$

考虑 $AC$ 对应的 $(i,j,k,i)$ 和 $AD$ 对应的 $(i,j,j,l)$，把要求写开，如下

$$
\begin{align*}
AC \ 的要求：a_j \mid a_i \ 且 \ a_i \mid a_k, \\
AD \ 的要求：a_i \mid a_j \ 且 \ a_j \mid a_l.
\end{align*}
$$ 

我们会发现这种要求都是以一个中间数 $x$ 为媒介，左右各是 $x$ 的因数和倍数。那么这种情况的答案就是

$$
\sum\limits cnt[x] \times fac[x] \times mul[x] \times 2.
$$

##### 最后回到 $A$。

$A$ 的计算就很简单了，只要 $i \neq k$ 且 $j \neq l$，这就相当于 $ABE$ 对应的 $(i,i,k,k)$ 的答案的平方，即

$$
\sum\limits (cnt[x] \times mul[x])^2.
$$

##### 综上所述。

最终答案

$$
\begin{align*}
ans =& \sum\limits_{x = 1}^{\max(a)}(cnt[x] \times mul[x])^2 + (cnt[x] \times mul[x]) \\
&- cnt[x] \times mul[x] \times mul[x] \\
&- cnt[x] \times fac[x] \times fac[x] \\
&- cnt[x] \times fac[x] \times mul[x] \\
&+ cnt[x] \times (cnt[x] - 1).
\end{align*}
$$

其中 $mul[x]$ 和 $fac[x]$ 的含义已经在上面有所解释。

注意这题需要开 `__int128`。

#### 时间复杂度 $\mathcal{O}(n + V\log V)$

- 其中 $V = \max(a)$。

#### C++ Code

```cpp
#include <bits/stdc++.h>

using i64 = long long;
using u64 = unsigned long long;
using u32 = unsigned;
using i80 = __int128_t;
using u80 = unsigned __int128_t;
using f64 = double;
using f80 = long double;

constexpr i64 inf = 1E18;

template<class T>
std::istream &operator>>(std::istream &is, std::vector<T> &a) {
    for (auto &x: a) {
        is >> x;
    }
    return is;
}
std::ostream &operator<<(std::ostream &os, const i80 &a) {
    if (a <= inf) {
        return os << i64(a);
    }
    return os << i64(a / inf) << std::setw(18) << std::setfill('0') << i64(a % inf);
}
i80 power(i80 a, i64 b) {
    i80 res = 1;
    for ( ; b; b /= 2, a *= a) {
        if (b & 1) {
            res *= a;
        }
    }
    return res;
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    
    int n;
    std::cin >> n;

    std::vector<int> a(n);
    std::cin >> a;

    int max = *std::max_element(a.begin(), a.end()) + 1;

    std::vector<int> cnt(max);
    for (int x: a) {
        cnt[x]++;
    }

    std::vector<int> fac(max);
    std::vector<int> mul(max);

    i64 res = 0;
    for (int x = 1; x < max; x++) {
        if (cnt[x] > 0) {
            fac[x] += cnt[x] - 1;
            mul[x] += cnt[x] - 1;
            for (int y = x + x; y < max; y += x) {
                fac[y] += cnt[x];
                mul[x] += cnt[y];
            }
            res += static_cast<i64>(cnt[x]) * mul[x];
        }
    }

    i80 ans = static_cast<i80>(res) * (res + 1);
    for (int x = 1; x < max; x++) {
        if (cnt[x] > 0) {
            ans -= static_cast<i80>(cnt[x]) * mul[x] * mul[x];
            ans -= static_cast<i80>(cnt[x]) * fac[x] * fac[x];
            ans -= static_cast<i80>(cnt[x]) * fac[x] * mul[x] * 2;
            ans += static_cast<i64>(cnt[x]) * (cnt[x] - 1);
        }
    }
    std::cout << ans << "\n";
    
    return 0;
}
```
