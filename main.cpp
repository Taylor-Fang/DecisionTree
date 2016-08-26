#include "DecisionTree.h"
#include <fstream>
#include <cstdlib>

using namespace std;

DataTable TrainSet;
DataTable TestSet;
ClassCollection TrainClassSet;
ClassCollection TestClassSet;
IndexCollection attrIndex;
pNode head,ihead;
TreeNode treeNode;

//前序遍历
void preorder(pNode head)
{
	if(head)
	{
		cout<<head->attrIndex<<endl;
		preorder(head->leftChild);
		preorder(head->rightChild);
	}
}

int main()
{
	ifstream trainDataIn;
	ifstream testDataIn;
	ofstream resultOut;
	Result res;

	trainDataIn.open("TrainData.txt");//训练集数据
	if(!trainDataIn.is_open())
	{
		cout<<"Can not open TrainData.txt !"<<endl;
		exit(EXIT_FAILURE);
	}

	for(unsigned int i = 0;i < 200000;i++)
	{
		int label;
		double featureValue;
		DataRow row;
		trainDataIn>>label;
		
		TrainClassSet.push_back(label);
		for(int j = 0;j < 14;j++)
		{
			trainDataIn>>featureValue;
			row.push_back(featureValue);
		
		}
		
		TrainSet.push_back(row);
		row.clear();
	}
	
	trainDataIn.close();
	cout<<"TrainData.txt has been read..."<<endl;

	testDataIn.open("TestData.txt");//测试集数据
	if(!testDataIn.is_open())
	{
		cout<<"Can not open TestData.txt !"<<endl;
		exit(EXIT_FAILURE);
	}

	for(unsigned int i = 0;i < 5747;i++)
	{
		int label;
		double featureValue;
		DataRow row;

		testDataIn>>label;
		TestClassSet.push_back(label);
		for(int j = 0 ;j < 14;j++)
		{
			testDataIn>>featureValue;
			row.push_back(featureValue);
		}
		TestSet.push_back(row);
		row.clear();
	}

	testDataIn.close();
	cout<<"TestData.txt has been read..."<<endl;

	for(int i = 0;i < 14;i++)
	{
		attrIndex.push_back(i);
	}

	resultOut.open("result.txt");
	head = DecisionTree::buildTree(TrainSet,attrIndex,TrainClassSet);
	
	DecisionTree::postPrune(head);
	

	res = DecisionTree::test(head,TestSet,TestClassSet);

	DecisionTree::removeTree(head);
	cout<<"the correct rate is : "<<res.rate<<endl;
	cout<<"there are : "<<res.count<<" samples"<<endl;

	TrainSet.clear();
	TrainClassSet.clear();
	TestSet.clear();
	TestClassSet.clear();

	return 0;
}
