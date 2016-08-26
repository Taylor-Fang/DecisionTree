#include "DecisionTree.h"

int bestIndex;//��ѷ�������

void DecisionTree::count(DataTable & samples, pCountCollection &sampleCount, IndexCollection & attrIndex,ClassCollection & classArray,DividerSet & dividerArray)
{
	sampleCount = new CountCollection [attrIndex.size()];
	int colCount = (int)attrIndex.size();//������(����)
	int valueNum = 2;//��Ϊ���������ԣ�ֻ�����ֽ��
	int rowNum = (int)classArray.size();//��������

	//����ͳ�ƽṹ����ʼ��
	for(int col = 0;col < colCount; col++)
	{
		for(int i = 0; i < valueNum; i++)
		{
			//sampleCount�洢ͳ�ƽ��
			//ÿ�����Դ洢4��ͳ�ƽ��
			/******************************************
			 *           * <���ŷָ�ֵ * >=���ŷָ�ֵ *
			 ******************************************
			 * [0]���-1 *             *              *
			 ******************************************
			 * [1]���1  *             *              *
			 ******************************************/
			sampleCount[col].push_back(new int [2]);//2��ʾֻ���������
			for(int j = 0;j < 2;j++)
				sampleCount[col][i][j] = 0;//��ʼ��
		}
	}

	//ͳ��
	for(int col = 0;col < colCount; col++)
	{
		double bestDivider = 0;//�������Ե����ŷָ�ֵ
		bestDivider = discrete(rowNum,col,classArray,samples,attrIndex[col]);//��ȡcol�����ŷָ�ֵ
		dividerArray.push_back(bestDivider);

		for(int rowIndex = 0;rowIndex < rowNum;rowIndex++)
		{
			//flag == 0��ʾС�ڷָ�ֵ flag == 1��ʾ���ڵ��ڷָ�ֵ
			int flag = 1;
			if(samples[rowIndex][col] < bestDivider)
				flag = 0;

			//m == 0��ʾ���-1,m == 1��ʾ���1
			int m = 0;
			if(classArray[rowIndex] == 1)
				m = 1;
			sampleCount[col][flag][m]++;
		}
	}
}

//��ȡcol�����Ե����ŷָ�ֵ
double DecisionTree::discrete(int rowNum, int colIndex, ClassCollection & classArray, DataTable & samples, int truecol)
{
	//�洢�������Ե�����ֵ
	double *attributeValue = new double [rowNum];

	//�洢����ֵ����Ӧ�����ֵ
	int *attributeLabel = new int [rowNum];
	for(int row = 0;row < rowNum;row++)
	{
		attributeValue[row] = samples[row][colIndex];
		attributeLabel[row] = classArray[row];
	}

	//����ð�����򷨣���Ҫ�Ľ���
	for(int row = 0; row < rowNum - 1; row++)
	{
		for(int j = rowNum - 1;j > row; j--)
		{
			if(attributeValue[j] < attributeValue[j-1])
			{
				double temp1 = attributeValue[j];
				attributeValue[j] = attributeValue[j-1];
				attributeValue[j-1] = temp1;
				int temp2 = attributeLabel[j];
				attributeLabel[j] = attributeLabel[j-1];
				attributeLabel[j-1] = temp2;
			}
		}
	}
	//little�洢С�ڷָ�ֵ��ÿ�����������large��֮
	//little[0]�洢���Ϊ-1��������little[1]�洢���Ϊ1��������largeͬ��
	int little[2],large[2];
	double bestDivider = 0.0;//bestDivider���ŷָ�ֵ
	double maxGainRatio = -1000;//maxGainRatio���������

	for(int row = 0;row < rowNum-1;row++)
	{
		if(attributeLabel[row] != attributeLabel[row+1])
		{
			for(int i = 0;i < 2; i++)
			{
				little[i] = 0;
				large[i] = 0;
			}
		
			for(int j = 0;j < row+1; j++)
			{
				if(attributeLabel[j] == -1)
					little[0]++;
				else
					little[1]++;
			}

			for(int j = row+1;j < rowNum; j++)
			{
				if(attributeLabel[j] == -1)
					large[0]++;
				else
					large[1]++;
			}
		}
		//��ȡ����������
		double infoGainRatio = gainRatio(rowNum,little,large);

		if(infoGainRatio > maxGainRatio)
		{
			bestDivider = attributeValue[row+1];
			maxGainRatio = infoGainRatio;
		}
	}
	delete [] attributeValue;
	delete [] attributeLabel;

	return bestDivider;
}

//������Ϣ��
//labelCount[0]��ʾ���-1����Ŀ��labelCount[1]��ʾ���1����Ŀ��allNum��ʾ����
double DecisionTree::entropy(int *labelCount)
{
	double iEntropy = 0.0;
	int allNum = labelCount[0]+labelCount[1];
	for(int i = 0; i < 2; i++)
	{
		double temp = ((double)labelCount[i])/allNum;
		if(temp != 0.0)
			iEntropy -= temp*(log(temp)/log(2.0));
	}

	return iEntropy;
}

//����������
double DecisionTree::gainRatio(int allNum,int *little,int *large)
{
	double gain = 0.0;
	double splitInfo = 0.0;
	int labelCount[2] = {0,0};
	labelCount[0] = little[0] + large[0];
	labelCount[1] = little[1] + large[1];
	
	double sampleEntropy = entropy(labelCount);
	double entropy1 = (double(little[0]+little[1]))/allNum*entropy(little);
	double entropy2 = (double(large[0]+large[1]))/allNum*entropy(large);
	gain = sampleEntropy - entropy1 - entropy2;

	double splitInfo1 = -(double(little[0]+little[1]))/allNum*(log(double(little[0]+little[1])/allNum)/log(2.0));
	double splitInfo2 = -(double(large[0]+large[1]))/allNum*(log(double(large[0]+large[1])/allNum)/log(2.0));
	splitInfo = splitInfo1 + splitInfo2;

	return (gain/splitInfo);
}

//ѡ����ѷ�������
int DecisionTree::chooseBestAttribute(IndexCollection attrIndex, pCountCollection sampleCount)
{
	int bestIndex = 0;
	double maxGainRatio = -100.0;
	int allNum = sampleCount[0][0][0]+sampleCount[0][0][1]+sampleCount[0][1][0]+sampleCount[0][1][1];

	for(int colIndex = 0;colIndex < (int)attrIndex.size();colIndex++)
	{
		double gainR = gainRatio(allNum,sampleCount[colIndex][0],sampleCount[colIndex][1]);
		if(gainR > maxGainRatio)
		{
			maxGainRatio = gainR;
			bestIndex = colIndex;
		}
	}

	return bestIndex;
}

//�ж�����������Ƿ�һ��
bool DecisionTree::isSameResultValue(ClassCollection & classArray)
{
	int temp = classArray[0];
	bool same = true;
	for(int row = 0;row < classArray.size();row++)
	{
		if(classArray[row] != temp)
		{
			same = false;
			break;
		}
	}

	return same;
}

//����������
pNode DecisionTree::buildTree(DataTable & samples, IndexCollection attrIndex, ClassCollection & classArray)
{
	int rowCount = (int)samples.size();//������
	int colCount = (int)attrIndex.size();//������(�������������)

	//������Ϊ�������,���޸ģ�Ӧֱ�ӷ���
	if(samples.size() == 0)
	{
		pNode leafNode = new Node ;
		leafNode->isLeaf = true;
		leafNode->attrIndex = -1;//û�з������ԣ�����Ϊ-1
		leafNode->popularClass = 0;//��
		leafNode->bestDivider = -1;
		leafNode->count[0] = 0;
		leafNode->count[1] = 0;
		leafNode->leftChild = nullptr;
		leafNode->rightChild = nullptr;

		return leafNode;
	}

	//�����������ͬһ���
	if(isSameResultValue(classArray))
	{
		int temp = classArray[0];
		pNode leafNode = new Node;
		leafNode->isLeaf = true;
		leafNode->attrIndex = -1;//��������ͬһ�����˲���Ҫ�ٷ���
		leafNode->popularClass = temp;
		leafNode->bestDivider = -1;//����Ҫ���ѣ�����Ҳ����Ҫ�ָ�ֵ
		leafNode->count[0] = rowCount;
		leafNode->count[1] = 0;
		leafNode->leftChild = nullptr;
		leafNode->rightChild = nullptr;

		return leafNode;
	}

	pNode iNode = new Node;
	int labelNum = 2;//���������
	int checkCount[2] = {0,0};
	for(int rowIndex = 0;rowIndex < rowCount;rowIndex++)
	{
		if(classArray[rowIndex] == -1)
			checkCount[0]++;
		else
			checkCount[1]++;
	}

	int maxCount = 0;
	int popularClass = 0;
	if(checkCount[0] > checkCount[1])
	{
		maxCount = checkCount[0];
		popularClass = -1;
	}
	else
	{
		maxCount = checkCount[1];
		popularClass = 1;
	}

	iNode->popularClass = popularClass;
	iNode->count[0] = maxCount;
	iNode->count[1] = rowCount - maxCount;

	//���û�пɷ��ѵ���������
	if(attrIndex.size() == 0)
	{
		iNode->isLeaf = true;
		iNode->attrIndex = -1;//û�пɷ��ѵ�����
		iNode->bestDivider = -1;
		iNode->leftChild = nullptr;
		iNode->rightChild = nullptr;
		return iNode;
	}

	//�ݹ飬��������
	pCountCollection sampleCount;
	DividerSet dividerArray;
	count(samples,sampleCount,attrIndex,classArray,dividerArray);
	
	bestIndex = chooseBestAttribute(attrIndex,sampleCount);
	for(int colIndex = 0;colIndex < colCount;colIndex++)
		for(int i = 0;i < 2;i++)
			delete [] sampleCount[colIndex][i]; 
	delete [] sampleCount;

	int bestAttribute = attrIndex[bestIndex];
	
	iNode->isLeaf = false;
	iNode->attrIndex = bestAttribute;
	iNode->bestDivider = dividerArray[bestIndex];
	int valueN = 2;//�������ԣ��ʷ���Ϊ��������
	iNode->leftChild = new Node;
	iNode->rightChild = new Node;
	DataTable *iSamples = new DataTable [valueN];
	ClassCollection *iclassArray = new ClassCollection [valueN];
	DataRow row;
	for(int rowIndex = 0;rowIndex < rowCount;rowIndex++)
	{
		int flag = 1;
		if(samples[rowIndex][bestIndex] < dividerArray[bestIndex])
			flag = 0;
		for(int colIndex = 0;colIndex < colCount;colIndex++)
		{
			if(colIndex != bestIndex)
				row.push_back(samples[rowIndex][colIndex]);
		}
		iSamples[flag].push_back(row);
		iclassArray[flag].push_back(classArray[rowIndex]);
		row.clear();
	}
	dividerArray.clear();

	IndexCollection newAttrIndex;
	for(int i = 0; i < (int)attrIndex.size(); i++)
	{
		if(i != bestIndex)
			newAttrIndex.push_back(attrIndex[i]);
	}
	iNode->leftChild = buildTree(iSamples[0], newAttrIndex,iclassArray[0]);
	iNode->rightChild = buildTree(iSamples[1], newAttrIndex,iclassArray[1]);
	
	delete [] iSamples;
	delete [] iclassArray;

	return iNode;
}

//ͳ������Ҷ�ӽڵ����,�������
int DecisionTree::leafCount(pNode & iNode)
{
	if(iNode == nullptr)
		return 0;
	if((iNode->leftChild == nullptr) && (iNode->rightChild == nullptr))
		return 1;

	return (leafCount(iNode->leftChild)+leafCount(iNode->rightChild));
}

//ͳ������Ҷ�ӽڵ��д���������
int DecisionTree::errCount(pNode &iNode)
{
	if(iNode == nullptr)
		return 0;
	if((iNode->leftChild == nullptr) && (iNode->rightChild == nullptr))
		return iNode->count[1];

	return (errCount(iNode->leftChild)+errCount(iNode->rightChild));
}

//���ۼ�֦��(��Ҫ�޸�)
bool DecisionTree::postPrune(pNode &iNode)
{
	if(iNode->isLeaf)
		return false;
	int leafNum = 0;//ͳ��Ҷ�ӽڵ����Ŀ
	int errNum = 0; //ͳ��Ҷ�ӽڵ��д���������
	leafNum = leafCount(iNode);
	errNum = errCount(iNode);
	double e = (double)(errNum+(double)leafNum/2)/(iNode->count[0]+iNode->count[1]);//������
	//E_subtree = Ҷ�ӽڵ��д����ܺ� + 0.5 * Ҷ�ӽڵ���
	double E_subtree = errNum + (double)leafNum/2;
	//E_node = �ýڵ�Ĵ����� + 0.5
	double E_node = iNode->count[1] + 0.5;
	//��׼��var = sqrt(N*e(1-e))
	double var = sqrt(E_subtree*(1-e));

	if((var+E_subtree) >= E_node)
	{
		Node * newNode = new Node;
		newNode->isLeaf = true;
		newNode->bestDivider = -1;
		newNode->attrIndex = -1;
		newNode->count[0] = iNode->count[0];
		newNode->count[1] = iNode->count[1];
		newNode->popularClass = iNode->popularClass;
		newNode->leftChild = nullptr;
		newNode->rightChild = nullptr;

		//ɾ��ԭ��������
		delete iNode;
		iNode = newNode;
		return true;
	}
	else
	{
		postPrune(iNode->leftChild);
		postPrune(iNode->rightChild);
	}
}

void DecisionTree::removeTree(pNode &head)
{
	int valueN = 2;
	if(head->isLeaf)
	{
		delete head;
		return;
	}
	removeTree(head->leftChild);
	removeTree(head->rightChild);

	delete head;
}

Result DecisionTree::test(pNode & treeNode,DataTable & testSet,ClassCollection & testClass)
{
	Result res;
	res.count = testSet.size();
	int correctCount = 0;

	//�������Ա�
	for(int row = 0;row < testSet.size();row++)
	{
		bool flag = todotest(testSet[row],treeNode,testClass[row]);
		if(flag == true)
			correctCount++;
	}

	res.rate = (double)correctCount/res.count;
	
	return res;
}

bool DecisionTree::todotest(DataRow & row,pNode & treeNode,int label)
{
	if(treeNode->isLeaf)
	{
		if(treeNode->popularClass == label)
			return true;
		else
			return false;
	}
	else
	{
		int attrIndex = treeNode->attrIndex;
		bool flag = false;
		if(row[attrIndex] < treeNode->bestDivider)
		    flag = todotest(row,treeNode->leftChild,label);
		else
			flag = todotest(row,treeNode->rightChild,label);
		return flag;
	}
}