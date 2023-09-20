#include "CommonLib/CommonDef.h"



#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <vector>



class RandomForestClassfier
{
	std::vector<int>  indices_qt_mtt_128_128 = { 6,10,24,18,8,27,2,21,34,0,35,14,25,32,4,1,3,37,11,20,29,13 };
	std::vector<int>  indices_qt_mtt_64_64 = {19,11,39,33,13,32,5,34,36,9,24,12,4,10,37};
	std::vector<int>  indices_qt_mtt_32_32 = {10, 34, 8, 39, 14, 13, 27, 12, 7, 2, 37};
	std::vector<int>  indices_qt_mtt_16_16 = { 27, 23, 33, 2, 11, 1, 21, 32, 4, 7};
	std::vector<int>  indices_hor_ver_8_8 = { 28, 24, 33, 3, 39, 0, 21, 10, 38, 5, 25, 1, 19, 4, 13, 36, 27, 14, 12, 11, 20, 17, 15, 35};
	std::vector<int>  indices_hor_ver_8_16 = {30, 13, 8, 5, 3, 31, 14, 36, 17, 20, 10, 25, 39, 32};
	std::vector<int>  indices_hor_ver_8_32 = { 33, 32, 6, 22, 7, 12, 20, 39, 9, 16, 31, 29, 34, 3, 2, 5 };
	std::vector<int>  indices_hor_ver_8_64 = { 22, 12, 3, 32, 34, 13, 2, 15, 18, 37 };
	std::vector<int>  indices_hor_ver_16_8 = { 12, 37, 19, 33, 6, 25, 36, 11, 28, 23, 0, 34, 9, 20, 5, 18, 29, 26, 4, 17, 10, 8, 39 };
	std::vector<int>  indices_hor_ver_16_16 = { 35, 30, 3, 9, 16, 2, 13, 33, 14, 18, 38, 29, 4, 6, 8, 36, 0, 7, 32, 24};
	std::vector<int>  indices_hor_ver_16_32 = { 22, 2, 27, 12, 29, 5, 37, 11, 10, 21 };
	std::vector<int>  indices_hor_ver_16_64 = { 22, 6, 4, 16, 0, 1, 10, 19, 32, 25, 18, 9, 27, 17, 28, 34, 13, 37, 39, 35};
	std::vector<int>  indices_hor_ver_32_8 = { 19, 20, 12, 17, 10, 16, 0, 25, 15, 22};
	std::vector<int>  indices_hor_ver_32_16 = { 11, 21, 39, 24, 35, 0, 13, 14, 28, 22, 25, 34, 2, 10, 1, 12};
	std::vector<int>  indices_hor_ver_32_32 = { 20, 13, 21, 29, 19, 34, 8, 9, 22, 32, 30, 16, 1, 35, 14};
	std::vector<int>  indices_hor_ver_32_64 = { 7, 24, 26, 15, 30, 9, 13, 22, 3, 0, 17, 18, 29, 11, 10, 16 };
	std::vector<int>  indices_hor_ver_64_8 = { 37, 1, 12, 38, 16, 11, 34, 20, 13, 0, 32 };
	std::vector<int>  indices_hor_ver_64_16 = { 27, 38, 25, 5, 16, 0, 21, 36, 22, 2, 3, 12};
	std::vector<int>  indices_hor_ver_64_32 = { 28, 30, 14, 11, 13, 29, 5, 20, 10, 4, 24, 34, 19 };
	std::vector<int>  indices_hor_ver_64_64 = { 8, 23, 1, 20, 3, 10, 13, 31, 28, 5, 2, 18 };
	std::vector<int>  indices_hor_ver_128_128 = { 38, 39, 30, 19, 20, 28, 22, 5, 18, 25, 31, 17, 8, 9, 24, 23, 2, 0, 7, 21, 6};


public:

	double predictQTMTT(float features[34], int wd, int ht)
	{
		switch (wd)
		{
		case 16:
		{
			switch (ht)
			{
			case 16:
				return predictQTMTT_16_16(features, indices_qt_mtt_16_16);
			default:
				return 0.5;
			}
		};
		case 32:
		{
			switch (ht)
			{
			case 32:
				return predictQTMTT_32_32(features, indices_qt_mtt_32_32);
			default:
				return 0.5;
			}
		};
		case 64:
		{
			switch (ht)
			{
			case 64:
				return predictQTMTT_64_64(features, indices_qt_mtt_64_64);
			default:
				return 0.5;
			}
		};
		case 128:
		{
			switch (ht)
			{
			case 128:
				return predictQTMTT_128_128(features, indices_qt_mtt_128_128);
			default:
				return 0.5;
			}
		};
		default:
			return 0.5;
		}
	};

	double predictHorVer(float features[45], int wd, int ht)
	{
		switch (wd)
		{
		case 8:
		{
			switch (ht)
			{
			case 8:
				return predictHorVer_8_8(features, indices_hor_ver_8_8);
			case 16:
				return predictHorVer_8_16(features, indices_hor_ver_8_16);
			case 32:
				return predictHorVer_8_32(features, indices_hor_ver_8_32);
			case 64:
				return predictHorVer_8_64(features, indices_hor_ver_8_64);
			default:
				return 0.5;
			}
		};
		case 16:
		{
			switch (ht)
			{
			case 8:
				return predictHorVer_16_8(features, indices_hor_ver_16_8);
			case 16:
				return predictHorVer_16_16(features, indices_hor_ver_16_16);
			case 32:
				return predictHorVer_16_32(features, indices_hor_ver_16_32);
			case 64:
				return predictHorVer_16_64(features, indices_hor_ver_16_64);
			default:
				return 0.5;
			}
		};
		case 32:
		{
			switch (ht)
			{
			case 8:
				return predictHorVer_32_8(features, indices_hor_ver_32_8);
			case 16:
				return predictHorVer_32_16(features, indices_hor_ver_32_16);
			case 32:
				return predictHorVer_32_32(features, indices_hor_ver_32_32);
			case 64:
				return predictHorVer_32_64(features, indices_hor_ver_32_64);
			default:
				return 0.5;
			}
		};
		case 64:
		{
			switch (ht)
			{
			case 8:
				return predictHorVer_64_8(features, indices_hor_ver_64_8);
			case 16:
				return predictHorVer_64_16(features, indices_hor_ver_64_16);
			case 32:
				return predictHorVer_64_32(features, indices_hor_ver_64_32);
			case 64:
				return predictHorVer_64_64(features, indices_hor_ver_64_64);
			default:
				return 0.5;
			}
		};
		case 128:
		{
			switch (ht)
			{
			case 128:
				return predictHorVer_128_128(features, indices_hor_ver_128_128);
			default:
				return 0.5;
			}
		};
		default:
			return 0.5;
		}
	};


private:
	double predictQTMTT_16_16(float features[], std::vector<int> indices);
	double predictQTMTT_32_32(float features[], std::vector<int> indices);
	double predictQTMTT_64_64(float features[], std::vector<int> indices);
	double predictQTMTT_128_128(float features[], std::vector<int> indices);

	double predictHorVer_8_8(float features[], std::vector<int> indices);
	double predictHorVer_16_8(float features[], std::vector<int> indices);
	double predictHorVer_8_16(float features[], std::vector<int> indices);
	double predictHorVer_16_16(float features[], std::vector<int> indices);
	double predictHorVer_32_8(float features[], std::vector<int> indices);
	double predictHorVer_8_32(float features[], std::vector<int> indices);
	double predictHorVer_32_16(float features[], std::vector<int> indices);
	double predictHorVer_16_32(float features[], std::vector<int> indices);
	double predictHorVer_32_32(float features[], std::vector<int> indices);
	double predictHorVer_64_32(float features[], std::vector<int> indices);
	double predictHorVer_32_64(float features[], std::vector<int> indices);
	double predictHorVer_64_8(float features[], std::vector<int> indices);
	double predictHorVer_8_64(float features[], std::vector<int> indices);
	double predictHorVer_64_16(float features[], std::vector<int> indices);
	double predictHorVer_16_64(float features[], std::vector<int> indices);
	double predictHorVer_64_64(float features[], std::vector<int> indices);
	double predictHorVer_128_128(float features[], std::vector<int> indices);

};

