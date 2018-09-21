# 决策树
决策树（Decision Tree）算法是一种基本的分类与回归方法，是最经常使用的数据挖掘算法之一。

决策树模型呈树形结构，在分类问题中，表示基于特征对实例进行分类的过程。它可以认为是 if-then 规则的集合，也可以认为是定义在特征空间与类空间上的条件概率分布。

决策树学习通常包括 3 个步骤：特征选择、决策树的生成和决策树的修剪。

## 定义

分类决策树模型是一种描述对实例进行分类的树形结构。

决策树由**结点（node）**和**有向边（directed edge）**组成。

结点有两种类型：**内部结点（internal node）**和**叶结点（leaf node）**。内部结点表示一个特征或属性(features)，叶结点表示一个类(labels)。

用决策树对需要测试的实例进行分类：从根节点开始，对实例的某一特征进行测试，根据测试结果，将实例分配到其子结点；这时，每一个子结点对应着该特征的一个取值。如此递归地对实例进行测试并分配，直至达到叶结点。最后将实例分配到叶结点的类中。

## 原理

### 概念

- **熵（entropy）**：熵指的是体系的混乱的程度，在不同的学科中也有引申出的更为具体的定义，是各领域十分重要的参量。

- **信息熵（information entropy）**：是一种信息的度量方式，表示信息的混乱程度，也就是说：信息越有序，信息熵越低。假定当前样本集合$D$中第$k$类样本所占的比例为 $p_k(k=1,2,...,|y|)$，则$D$的信息熵定义为
  $$
  Ent(D)=-\sum_{k=1}^{|y|}p_k\log_2p_k
  $$
  $Ent(D)$的值越小，则$D$的纯度越高。

- **信息增益（information gain）**：
  $$
  Gain(D,a)=Ent(D)-\sum_{v=1}^V\frac{|D^v|}{|D|}Ent(D^v)
  $$

  [^D]: 样本集

  [^a]: 离散属性，有V个可能的取值$\lbrace a^1,a^2,...,a^V \rbrace$