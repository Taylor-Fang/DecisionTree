#ifndef DECISIONTREE_H_
#define DECISIONTREE_H_

#include "resource.h"
#include <cmath>

typedef struct result
{
	//��������
	int count;
	//��ȷ��
	double rate;
} Result;

//���������ڸó����в�δʹ��
typedef struct treenode
{
	//�Ƿ���Ҷ�ӽڵ�
	bool isLeaf;
	//������������
	int attrIndex;
	//���ŷָ�ֵ
	double bestDivider;
	//���
	int label;
	//��������
	treenode * leftChild;
	treenode * rightChild;
}TreeNode;

typedef struct node
{
	//ָʾ�Ƿ���Ҷ�ӽڵ�
	bool isLeaf;
	//ָ��������Ե�����
	int attrIndex;
	//���ŷָ�ֵ
	double bestDivider;
	//[0]��¼��������������[1]��¼���ٵ�
	int count[2];
	//ָ�����������ձ�ֵ������
	int popularClass;
	//��һ���ڵ�
	node * leftChild;//������ <���ŷָ�ֵ
	node * rightChild;//������ >=���ŷָ�ֵ
}Node, *pNode;


//DecisionTree�еĳ�ԱΪstatic,�������ַǾ�̬��Ա���ñ������ض�������ԵĴ���
class DecisionTree
{
private:
	//countͳ�Ƹ������Ե�����
	//samples����������
	//sampleCount���洢ͳ�ƽ��
	//attrIndex��������������
	//classArray�����������
	//dividerArray���洢ÿ�����Ե����ŷָ�ֵ
	static void count(DataTable & samples, pCountCollection &sampleCount, IndexCollection & attrIndex,ClassCollection & classArray,DividerSet & dividerArray);
	//discrete��ȡ���Ե����ŷָ�ֵ
	//rowNum����������
	//colIndex��Ҫ�жϵ��������������е�����
	//classArray�����������
	//samples����������
	//truecol��=attrIndex[colIndex]
	static double discrete(int rowNum, int colIndex, ClassCollection & classArray, DataTable & samples, int truecol);
	//gainRatio����������
	//allNum����������
	//little���洢<=�ָ�ֵ���������ͳ�ƽ��
	//large���洢>�ָ�ֵ���������ͳ�ƽ��
	static double gainRatio(int allNum,int *little,int *large);
	//entropy������Ϣ��
	//labelCount��[0]�洢���-1��������[1]�洢���1������
	static double entropy(int *labelCount);
	//chooseBestAttributeѡ����ѵķ�������
	//attrIndex��������������
	//sampleCount���洢ͳ�ƽ��
	static int chooseBestAttribute(IndexCollection attrIndex, pCountCollection sampleCount);
	//isSameResultValue�����Ƿ���ͬһ���
	static bool isSameResultValue(ClassCollection & classArray);
	//todotest��row��Ԥ��������֪���label�Ƿ���ͬ
	static bool todotest(DataRow & row,pNode & treeNode,int label);
	//leafCoutͳ��������Ҷ�ӽڵ���
	static int leafCount(pNode &iNode);
	//errCountͳ������Ҷ�ӽڵ��е������
	static int errCount(pNode &iNode);
public:
	//buildTree����
	//samples������
	//classArrray��������𼯺�
	//attrIndex��������������
	static pNode buildTree(DataTable & samples, IndexCollection attrIndex,ClassCollection & classArray);
	//��֦�����ñ��ۼ�֦��
	static bool postPrune(pNode &iNode);
	//removeTree��ɾ����������������ͷ���Դ
	static void removeTree(pNode &head);
	//test����Ԥ��
	//treeNode���������ĸ��ڵ�
	//test����������
	//testClass������������𼯺�
	static Result test(pNode & treeNode,DataTable & testSet,ClassCollection & testClass);
};

#endif