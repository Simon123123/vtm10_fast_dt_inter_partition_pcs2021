#include "CommonLib/CommonDef.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <vector>

class RandomForestClassfier
{
	std::vector<int>  indices_qt_mtt_128_128 = { 4,9,30,3,34,5,12,39,8,36,33,15,16,23,29,19,14,32,35,17,27,13,28,0,10 };
	std::vector<int>  indices_qt_mtt_64_64 = { 11,35,13,8,9,16,15,10,26,20,5,14,23 };
	std::vector<int>  indices_qt_mtt_32_32 = { 39,36,1,34,8,24,35,2,9,15,16 };
	std::vector<int>  indices_qt_mtt_16_16 = { 24,15,32,5,22,26,39,0,27,31,16 };
	std::vector<int>  indices_hor_ver_8_8 = { 11,33,0,4,7,20,6,35,21,13,19,27,32,30,8,17,15,34,1,5,16,38,9,18,28,12,22,2,23 };
	std::vector<int>  indices_hor_ver_8_16 = { 2,18,29,35,5,30,27,39,20,12,6,17,24,10,22,25,14,15,23,13,11,36,38,4,19,34,8 };
	std::vector<int>  indices_hor_ver_8_32 = { 32,23,16,9,30,8,27,15,18,1,17,28,6,2,39,26,3,7,29,13,11 };
	std::vector<int>  indices_hor_ver_8_64 = { 12,22,30,20,21,3,16,17,1,29,14,32,34 };
	std::vector<int>  indices_hor_ver_16_8 = { 14,23,31,18,22,35,17,21,4,39,38,37,1,0,29,12,7 };
	std::vector<int>  indices_hor_ver_16_16 = { 38,28,35,4,7,12,33,26,6,31,21,17,18,0,20,19,2 };
	std::vector<int>  indices_hor_ver_16_32 = { 28,5,10,13,35,14,23,34,1,7,8 };
	std::vector<int>  indices_hor_ver_16_64 = { 35,14,9,30,15,37,33,34,22,36,3,32,11,28,23,17,24 };
	std::vector<int>  indices_hor_ver_32_8 = { 26,28,35,0,17,29,18,15,38,25,16,39,37,27,36,13,9,22,21,7,8,33,23,4 };
	std::vector<int>  indices_hor_ver_32_16 = { 0,23,2,31,14,20,25,27,16,39,18,32,9,34 };
	std::vector<int>  indices_hor_ver_32_32 = { 4,7,26,29,13,37,34,30,38,39,31 };
	std::vector<int>  indices_hor_ver_32_64 = { 29,2,27,15,30,14,28,32,9,19,21,5,11 };
	std::vector<int>  indices_hor_ver_64_8 = { 9,0,5,27,12,4,34,19,21,26,2,33,10,37 };
	std::vector<int>  indices_hor_ver_64_16 = { 3,8,28,38,0,31,21,33,32,13,4,37,14,7,9,16,27,18,12,24,39,11,2,29,25,15 };
	std::vector<int>  indices_hor_ver_64_32 = { 30,39,18,17,23,13,21,38,9,7,31,36,14,15,16,22,25 };
	std::vector<int>  indices_hor_ver_64_64 = { 7,21,34,35,6,25,2,13,37,1,0,27,5,26,4 };
	std::vector<int>  indices_hor_ver_128_128 = { 15,32,12,22,25,35,24,31,14,23,11 };


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

