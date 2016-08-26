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

//Ç°Ðò±éÀú
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

	trainDataIn.open("TrainData.txt");
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
		//cout<<label<<" ";
		TrainClassSet.push_back(label);
		for(int j = 0;j < 14;j++)
		{
			trainDataIn>>featureValue;
			row.push_back(featureValue);
		//	cout<<featureValue<<" ";
		}
		//cout<<endl;
		TrainSet.push_back(row);
		row.clear();
	}
	/*for(int i = 0;i<124;i++)
	{
		cout<<TrainClassSet[i]<<" ";
		for(int j = 0;j< 6;j++)
		{
			cout<<TrainSet[i][j]<<" ";
		}
		cout<<endl;
	}*/
	trainDataIn.close();
	cout<<"TrainData.txt has been read..."<<endl;

	testDataIn.open("TestData.txt");
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
	//preorder(head);
	//while(DecisionTree::postPrune(head));
	//cout<<"***************"<<endl;
	DecisionTree::postPrune(head);
	//preorder(head);
	//cout<<"prune..."<<endl;

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
