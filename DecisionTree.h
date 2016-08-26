#ifndef DECISIONTREE_H_
#define DECISIONTREE_H_

#include "resource.h"
#include <cmath>

typedef struct result
{
	//样本总数
	int count;
	//正确率
	double rate;
} Result;

//规则树，在该程序中并未使用
typedef struct treenode
{
	//是否是叶子节点
	bool isLeaf;
	//分裂属性索引
	int attrIndex;
	//最优分割值
	double bestDivider;
	//类别
	int label;
	//左右子树
	treenode * leftChild;
	treenode * rightChild;
}TreeNode;

typedef struct node
{
	//指示是否是叶子节点
	bool isLeaf;
	//指向分裂属性的索引
	int attrIndex;
	//最优分割值
	double bestDivider;
	//[0]记录最多的类别的数量，[1]记录较少的
	int count[2];
	//指向结果属性中普遍值的索引
	int popularClass;
	//下一个节点
	node * leftChild;//左子树 <最优分割值
	node * rightChild;//右子树 >=最优分割值
}Node, *pNode;


//DecisionTree中的成员为static,否则会出现非静态成员引用必须与特定对象相对的错误
class DecisionTree
{
private:
	//count统计各个属性的数据
	//samples：样本数据
	//sampleCount：存储统计结果
	//attrIndex：属性索引集合
	//classArray：类别列数据
	//dividerArray：存储每个属性的最优分割值
	static void count(DataTable & samples, pCountCollection &sampleCount, IndexCollection & attrIndex,ClassCollection & classArray,DividerSet & dividerArray);
	//discrete获取属性的最优分割值
	//rowNum：样本数量
	//colIndex：要判断的属性在样本表中的索引
	//classArray：类别数据列
	//samples：样本数据
	//truecol：=attrIndex[colIndex]
	static double discrete(int rowNum, int colIndex, ClassCollection & classArray, DataTable & samples, int truecol);
	//gainRatio计算增益率
	//allNum：数据数量
	//little：存储<=分割值的类别数量统计结果
	//large：存储>分割值的类别数量统计结果
	static double gainRatio(int allNum,int *little,int *large);
	//entropy计算信息熵
	//labelCount：[0]存储类别-1的数量，[1]存储类别1的数量
	static double entropy(int *labelCount);
	//chooseBestAttribute选择最佳的分列属性
	//attrIndex：属性索引集合
	//sampleCount：存储统计结果
	static int chooseBestAttribute(IndexCollection attrIndex, pCountCollection sampleCount);
	//isSameResultValue样本是否是同一类别
	static bool isSameResultValue(ClassCollection & classArray);
	//todotest：row的预测结果与已知结果label是否相同
	static bool todotest(DataRow & row,pNode & treeNode,int label);
	//leafCout统计子树的叶子节点数
	static int leafCount(pNode &iNode);
	//errCount统计子树叶子节点中的误差数
	static int errCount(pNode &iNode);
public:
	//buildTree建树
	//samples：样本
	//classArrray：样本类别集合
	//attrIndex：属性索引集合
	static pNode buildTree(DataTable & samples, IndexCollection attrIndex,ClassCollection & classArray);
	//剪枝，采用悲观剪枝法
	static bool postPrune(pNode &iNode);
	//removeTree：删除树，程序结束后释放资源
	static void removeTree(pNode &head);
	//test进行预测
	//treeNode：决策树的根节点
	//test：测试样本
	//testClass：测试样本类别集合
	static Result test(pNode & treeNode,DataTable & testSet,ClassCollection & testClass);
};

#endif