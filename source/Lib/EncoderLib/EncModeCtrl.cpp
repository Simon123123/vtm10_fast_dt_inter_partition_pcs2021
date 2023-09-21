/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2020, ITU/ISO/IEC
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *  * Neither the name of the ITU/ISO/IEC nor the names of its contributors may
 *    be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */

/** \file     EncModeCtrl.cpp
    \brief    Encoder controller for trying out specific modes
*/

#include "EncModeCtrl.h"

#include "AQp.h"
#include "RateCtrl.h"

#include "CommonLib/RdCost.h"
#include "CommonLib/CodingStructure.h"
#include "CommonLib/Picture.h"
#include "CommonLib/UnitTools.h"

#include "CommonLib/dtrace_next.h"

#include <cmath>


#if COLLECT_DATASET
void WriteFormatted_features ( FILE* f, const char * format, ... )
{
  va_list args;
  va_start ( args, format );
  vfprintf ( f, format, args );
  fflush( f );
  va_end ( args );
}
#endif


void EncModeCtrl::init( EncCfg *pCfg, RateCtrl *pRateCtrl, RdCost* pRdCost )
{
  m_pcEncCfg      = pCfg;
  m_pcRateCtrl    = pRateCtrl;
  m_pcRdCost      = pRdCost;
  m_fastDeltaQP   = false;
#if SHARP_LUMA_DELTA_QP
  m_lumaQPOffset  = 0;

  initLumaDeltaQpLUT();
#endif
}

bool EncModeCtrl::tryModeMaster( const EncTestMode& encTestmode, const CodingStructure &cs, Partitioner& partitioner )
{
#if ENABLE_SPLIT_PARALLELISM
  if( m_ComprCUCtxList.back().isLevelSplitParallel )
  {
    if( !parallelJobSelector( encTestmode, cs, partitioner ) )
    {
      return false;
    }
  }
#endif
  return tryMode( encTestmode, cs, partitioner );
}

void EncModeCtrl::setEarlySkipDetected()
{
  m_ComprCUCtxList.back().earlySkip = true;
}

void EncModeCtrl::xExtractFeatures( const EncTestMode encTestmode, CodingStructure& cs )
{
  CHECK( cs.features.size() < NUM_ENC_FEATURES, "Features vector is not initialized" );

  cs.features[ENC_FT_DISTORTION     ] = double( cs.dist              );
  cs.features[ENC_FT_FRAC_BITS      ] = double( cs.fracBits          );
  cs.features[ENC_FT_RD_COST        ] = double( cs.cost              );
  cs.features[ENC_FT_ENC_MODE_TYPE  ] = double( encTestmode.type     );
  cs.features[ENC_FT_ENC_MODE_OPTS  ] = double( encTestmode.opts     );
}

bool EncModeCtrl::nextMode( const CodingStructure &cs, Partitioner &partitioner )
{
  m_ComprCUCtxList.back().lastTestMode = m_ComprCUCtxList.back().testModes.back();

  m_ComprCUCtxList.back().testModes.pop_back();

  while( !m_ComprCUCtxList.back().testModes.empty() && !tryModeMaster( currTestMode(), cs, partitioner ) )
  {
    m_ComprCUCtxList.back().testModes.pop_back();
  }

  return !m_ComprCUCtxList.back().testModes.empty();
}

EncTestMode EncModeCtrl::currTestMode() const
{
  return m_ComprCUCtxList.back().testModes.back();
}

EncTestMode EncModeCtrl::lastTestMode() const
{
  return m_ComprCUCtxList.back().lastTestMode;
}

bool EncModeCtrl::anyMode() const
{
  return !m_ComprCUCtxList.back().testModes.empty();
}

void EncModeCtrl::setBest( CodingStructure& cs )
{
  if( cs.cost != MAX_DOUBLE && !cs.cus.empty() )
  {
    m_ComprCUCtxList.back().bestCS = &cs;
    m_ComprCUCtxList.back().bestCU = cs.cus[0];
    m_ComprCUCtxList.back().bestTU = cs.cus[0]->firstTU;
    m_ComprCUCtxList.back().lastTestMode = getCSEncMode( cs );
  }
}

void EncModeCtrl::xGetMinMaxQP( int& minQP, int& maxQP, const CodingStructure& cs, const Partitioner &partitioner, const int baseQP, const SPS& sps, const PPS& pps, const PartSplit splitMode )
{
  if( m_pcEncCfg->getUseRateCtrl() )
  {
    minQP = m_pcRateCtrl->getRCQP();
    maxQP = m_pcRateCtrl->getRCQP();
    return;
  }

  const unsigned subdivIncr = (splitMode == CU_QUAD_SPLIT) ? 2 : (splitMode == CU_BT_SPLIT) ? 1 : 0;
  const bool qgEnable = partitioner.currQgEnable(); // QG possible at current level
  const bool qgEnableChildren = qgEnable && ((partitioner.currSubdiv + subdivIncr) <= cs.slice->getCuQpDeltaSubdiv()) && (subdivIncr > 0); // QG possible at next level
  const bool isLeafQG = (qgEnable && !qgEnableChildren);

  if( isLeafQG ) // QG at deepest level
  {
    int deltaQP = m_pcEncCfg->getMaxDeltaQP();
    minQP = Clip3( -sps.getQpBDOffset( CHANNEL_TYPE_LUMA ), MAX_QP, baseQP - deltaQP );
    maxQP = Clip3( -sps.getQpBDOffset( CHANNEL_TYPE_LUMA ), MAX_QP, baseQP + deltaQP );
  }
  else if( qgEnableChildren ) // more splits and not the deepest QG level
  {
    minQP = baseQP;
    maxQP = baseQP;
  }
  else // deeper than QG
  {
    minQP = cs.currQP[partitioner.chType];
    maxQP = cs.currQP[partitioner.chType];
  }
}


int EncModeCtrl::xComputeDQP( const CodingStructure &cs, const Partitioner &partitioner )
{
  Picture* picture    = cs.picture;
  unsigned uiAQDepth  = std::min( partitioner.currSubdiv/2, ( uint32_t ) picture->aqlayer.size() - 1 );
  AQpLayer* pcAQLayer = picture->aqlayer[uiAQDepth];

  double dMaxQScale   = pow( 2.0, m_pcEncCfg->getQPAdaptationRange() / 6.0 );
  double dAvgAct      = pcAQLayer->getAvgActivity();
  double dCUAct       = pcAQLayer->getActivity( cs.area.Y().topLeft() );
  double dNormAct     = ( dMaxQScale*dCUAct + dAvgAct ) / ( dCUAct + dMaxQScale*dAvgAct );
  double dQpOffset    = log( dNormAct ) / log( 2.0 ) * 6.0;
  int    iQpOffset    = int( floor( dQpOffset + 0.49999 ) );
  return iQpOffset;
}


#if SHARP_LUMA_DELTA_QP
void EncModeCtrl::initLumaDeltaQpLUT()
{
  const LumaLevelToDeltaQPMapping &mapping = m_pcEncCfg->getLumaLevelToDeltaQPMapping();

  if( !mapping.isEnabled() )
  {
    return;
  }

  // map the sparse LumaLevelToDeltaQPMapping.mapping to a fully populated linear table.

  int         lastDeltaQPValue = 0;
  std::size_t nextSparseIndex = 0;
  for( int index = 0; index < LUMA_LEVEL_TO_DQP_LUT_MAXSIZE; index++ )
  {
    while( nextSparseIndex < mapping.mapping.size() && index >= mapping.mapping[nextSparseIndex].first )
    {
      lastDeltaQPValue = mapping.mapping[nextSparseIndex].second;
      nextSparseIndex++;
    }
    m_lumaLevelToDeltaQPLUT[index] = lastDeltaQPValue;
  }
}

int EncModeCtrl::calculateLumaDQP( const CPelBuf& rcOrg )
{
  double avg = 0;

  // Get QP offset derived from Luma level
#if !WCG_EXT
  if( m_pcEncCfg->getLumaLevelToDeltaQPMapping().mode == LUMALVL_TO_DQP_AVG_METHOD )
#else
  CHECK( m_pcEncCfg->getLumaLevelToDeltaQPMapping().mode != LUMALVL_TO_DQP_AVG_METHOD, "invalid delta qp mode" );
#endif
  {
    // Use average luma value
    avg = (double) rcOrg.computeAvg();
  }
#if !WCG_EXT
  else
  {
    // Use maximum luma value
    int maxVal = 0;
    for( uint32_t y = 0; y < rcOrg.height; y++ )
    {
      for( uint32_t x = 0; x < rcOrg.width; x++ )
      {
        const Pel& v = rcOrg.at( x, y );
        if( v > maxVal )
        {
          maxVal = v;
        }
      }
    }
    // use a percentage of the maxVal
    avg = ( double ) maxVal * m_pcEncCfg->getLumaLevelToDeltaQPMapping().maxMethodWeight;
  }
#endif
  int lumaBD = m_pcEncCfg->getBitDepth(CHANNEL_TYPE_LUMA);
  int lumaIdxOrg = Clip3<int>(0, int(1 << lumaBD) - 1, int(avg + 0.5));
  int lumaIdx = lumaBD < 10 ? lumaIdxOrg << (10 - lumaBD) : lumaBD > 10 ? lumaIdxOrg >> (lumaBD - 10) : lumaIdxOrg;
  int QP = m_lumaLevelToDeltaQPLUT[lumaIdx];
  return QP;
}
#endif

#if ENABLE_SPLIT_PARALLELISM
void EncModeCtrl::copyState( const EncModeCtrl& other, const UnitArea& area )
{
  m_slice          = other.m_slice;
  m_fastDeltaQP    = other.m_fastDeltaQP;
  m_lumaQPOffset   = other.m_lumaQPOffset;
  m_runNextInParallel
                   = other.m_runNextInParallel;
  m_ComprCUCtxList = other.m_ComprCUCtxList;
}

#endif
void CacheBlkInfoCtrl::create()
{
  const unsigned numPos = MAX_CU_SIZE >> MIN_CU_LOG2;

  m_numWidths  = gp_sizeIdxInfo->numWidths();
  m_numHeights = gp_sizeIdxInfo->numHeights();

  for( unsigned x = 0; x < numPos; x++ )
  {
    for( unsigned y = 0; y < numPos; y++ )
    {
      m_codedCUInfo[x][y] = new CodedCUInfo**[m_numWidths];

      for( int wIdx = 0; wIdx < gp_sizeIdxInfo->numWidths(); wIdx++ )
      {
        if( gp_sizeIdxInfo->isCuSize( gp_sizeIdxInfo->sizeFrom( wIdx ) ) && x + ( gp_sizeIdxInfo->sizeFrom( wIdx ) >> MIN_CU_LOG2 ) <= ( MAX_CU_SIZE >> MIN_CU_LOG2 ) )
        {
          m_codedCUInfo[x][y][wIdx] = new CodedCUInfo*[gp_sizeIdxInfo->numHeights()];

          for( int hIdx = 0; hIdx < gp_sizeIdxInfo->numHeights(); hIdx++ )
          {
            if( gp_sizeIdxInfo->isCuSize( gp_sizeIdxInfo->sizeFrom( hIdx ) ) && y + ( gp_sizeIdxInfo->sizeFrom( hIdx ) >> MIN_CU_LOG2 ) <= ( MAX_CU_SIZE >> MIN_CU_LOG2 ) )
            {
              m_codedCUInfo[x][y][wIdx][hIdx] = new CodedCUInfo;
            }
            else
            {
              m_codedCUInfo[x][y][wIdx][hIdx] = nullptr;
            }
          }
        }
        else
        {
          m_codedCUInfo[x][y][wIdx] = nullptr;
        }
      }
    }
  }
}

void CacheBlkInfoCtrl::destroy()
{
  const unsigned numPos = MAX_CU_SIZE >> MIN_CU_LOG2;

  for( unsigned x = 0; x < numPos; x++ )
  {
    for( unsigned y = 0; y < numPos; y++ )
    {
      for( int wIdx = 0; wIdx < gp_sizeIdxInfo->numWidths(); wIdx++ )
      {
        if( m_codedCUInfo[x][y][wIdx] )
        {
          for( int hIdx = 0; hIdx < gp_sizeIdxInfo->numHeights(); hIdx++ )
          {
            if( m_codedCUInfo[x][y][wIdx][hIdx] )
            {
              delete m_codedCUInfo[x][y][wIdx][hIdx];
            }
          }

          delete[] m_codedCUInfo[x][y][wIdx];
        }
      }

      delete[] m_codedCUInfo[x][y];
    }
  }
}

void CacheBlkInfoCtrl::init( const Slice &slice )
{
  const unsigned numPos = MAX_CU_SIZE >> MIN_CU_LOG2;

  for( unsigned x = 0; x < numPos; x++ )
  {
    for( unsigned y = 0; y < numPos; y++ )
    {
      for( int wIdx = 0; wIdx < gp_sizeIdxInfo->numWidths(); wIdx++ )
      {
        if( m_codedCUInfo[x][y][wIdx] )
        {
          for( int hIdx = 0; hIdx < gp_sizeIdxInfo->numHeights(); hIdx++ )
          {
            if( m_codedCUInfo[x][y][wIdx][hIdx] )
            {
              memset( m_codedCUInfo[x][y][wIdx][hIdx], 0, sizeof( CodedCUInfo ) );
            }
          }
        }
      }
    }
  }

  m_slice_chblk = &slice;
#if ENABLE_SPLIT_PARALLELISM

  m_currTemporalId = 0;
#endif
}
#if ENABLE_SPLIT_PARALLELISM

void CacheBlkInfoCtrl::touch( const UnitArea& area )
{
  CodedCUInfo& cuInfo = getBlkInfo( area );
  cuInfo.temporalId = m_currTemporalId;
}

void CacheBlkInfoCtrl::copyState( const CacheBlkInfoCtrl &other, const UnitArea& area )
{
  m_slice_chblk = other.m_slice_chblk;

  m_currTemporalId = other.m_currTemporalId;

  if( m_slice_chblk->isIntra() ) return;

  const int cuSizeMask = m_slice_chblk->getSPS()->getMaxCUWidth() - 1;

  const int minPosX = ( area.lx() & cuSizeMask ) >> MIN_CU_LOG2;
  const int minPosY = ( area.ly() & cuSizeMask ) >> MIN_CU_LOG2;
  const int maxPosX = ( area.Y().bottomRight().x & cuSizeMask ) >> MIN_CU_LOG2;
  const int maxPosY = ( area.Y().bottomRight().y & cuSizeMask ) >> MIN_CU_LOG2;

  for( unsigned x = minPosX; x <= maxPosX; x++ )
  {
    for( unsigned y = minPosY; y <= maxPosY; y++ )
    {
      for( int wIdx = 0; wIdx < gp_sizeIdxInfo->numWidths(); wIdx++ )
      {
        const int width = gp_sizeIdxInfo->sizeFrom( wIdx );

        if( m_codedCUInfo[x][y][wIdx] && width <= area.lwidth() && x + ( width >> MIN_CU_LOG2 ) <= ( maxPosX + 1 ) )
        {
          for( int hIdx = 0; hIdx < gp_sizeIdxInfo->numHeights(); hIdx++ )
          {
            const int height = gp_sizeIdxInfo->sizeFrom( hIdx );

            if( gp_sizeIdxInfo->isCuSize( height ) && height <= area.lheight() && y + ( height >> MIN_CU_LOG2 ) <= ( maxPosY + 1 ) )
            {
              if( other.m_codedCUInfo[x][y][wIdx][hIdx]->temporalId > m_codedCUInfo[x][y][wIdx][hIdx]->temporalId )
              {
                *m_codedCUInfo[x][y][wIdx][hIdx] = *other.m_codedCUInfo[x][y][wIdx][hIdx];
                m_codedCUInfo[x][y][wIdx][hIdx]->temporalId = m_currTemporalId;
              }
            }
            else if( y + ( height >> MIN_CU_LOG2 ) > maxPosY + 1 )
            {
              break;;
            }
          }
        }
        else if( x + ( width >> MIN_CU_LOG2 ) > maxPosX + 1 )
        {
          break;
        }
      }
    }
  }
}
#endif

CodedCUInfo& CacheBlkInfoCtrl::getBlkInfo( const UnitArea& area )
{
  unsigned idx1, idx2, idx3, idx4;
  getAreaIdx( area.Y(), *m_slice_chblk->getPPS()->pcv, idx1, idx2, idx3, idx4 );

  return *m_codedCUInfo[idx1][idx2][idx3][idx4];
}

bool CacheBlkInfoCtrl::isSkip( const UnitArea& area )
{
  unsigned idx1, idx2, idx3, idx4;
  getAreaIdx( area.Y(), *m_slice_chblk->getPPS()->pcv, idx1, idx2, idx3, idx4 );

  return m_codedCUInfo[idx1][idx2][idx3][idx4]->isSkip;
}

char CacheBlkInfoCtrl::getSelectColorSpaceOption(const UnitArea& area)
{
  unsigned idx1, idx2, idx3, idx4;
  getAreaIdx(area.Y(), *m_slice_chblk->getPPS()->pcv, idx1, idx2, idx3, idx4);

  return m_codedCUInfo[idx1][idx2][idx3][idx4]->selectColorSpaceOption;
}

bool CacheBlkInfoCtrl::isMMVDSkip(const UnitArea& area)
{
  unsigned idx1, idx2, idx3, idx4;
  getAreaIdx(area.Y(), *m_slice_chblk->getPPS()->pcv, idx1, idx2, idx3, idx4);

  return m_codedCUInfo[idx1][idx2][idx3][idx4]->isMMVDSkip;
}

void CacheBlkInfoCtrl::setMv( const UnitArea& area, const RefPicList refPicList, const int iRefIdx, const Mv& rMv )
{
  if( iRefIdx >= MAX_STORED_CU_INFO_REFS ) return;

  unsigned idx1, idx2, idx3, idx4;
  getAreaIdx( area.Y(), *m_slice_chblk->getPPS()->pcv, idx1, idx2, idx3, idx4 );

  m_codedCUInfo[idx1][idx2][idx3][idx4]->saveMv [refPicList][iRefIdx] = rMv;
  m_codedCUInfo[idx1][idx2][idx3][idx4]->validMv[refPicList][iRefIdx] = true;
#if ENABLE_SPLIT_PARALLELISM

  touch( area );
#endif
}

bool CacheBlkInfoCtrl::getMv( const UnitArea& area, const RefPicList refPicList, const int iRefIdx, Mv& rMv ) const
{
  unsigned idx1, idx2, idx3, idx4;
  getAreaIdx( area.Y(), *m_slice_chblk->getPPS()->pcv, idx1, idx2, idx3, idx4 );

  if( iRefIdx >= MAX_STORED_CU_INFO_REFS )
  {
    rMv = m_codedCUInfo[idx1][idx2][idx3][idx4]->saveMv[refPicList][0];
    return false;
  }

  rMv = m_codedCUInfo[idx1][idx2][idx3][idx4]->saveMv[refPicList][iRefIdx];
  return m_codedCUInfo[idx1][idx2][idx3][idx4]->validMv[refPicList][iRefIdx];
}

void SaveLoadEncInfoSbt::init( const Slice &slice )
{
  m_sliceSbt = &slice;
}

void SaveLoadEncInfoSbt::create()
{
  int numSizeIdx = gp_sizeIdxInfo->idxFrom( SBT_MAX_SIZE ) - MIN_CU_LOG2 + 1;
  int numPosIdx = MAX_CU_SIZE >> MIN_CU_LOG2;

  m_saveLoadSbt = new SaveLoadStructSbt***[numPosIdx];

  for( int xIdx = 0; xIdx < numPosIdx; xIdx++ )
  {
    m_saveLoadSbt[xIdx] = new SaveLoadStructSbt**[numPosIdx];
    for( int yIdx = 0; yIdx < numPosIdx; yIdx++ )
    {
      m_saveLoadSbt[xIdx][yIdx] = new SaveLoadStructSbt*[numSizeIdx];
      for( int wIdx = 0; wIdx < numSizeIdx; wIdx++ )
      {
        m_saveLoadSbt[xIdx][yIdx][wIdx] = new SaveLoadStructSbt[numSizeIdx];
      }
    }
  }
}

void SaveLoadEncInfoSbt::destroy()
{
  int numSizeIdx = gp_sizeIdxInfo->idxFrom( SBT_MAX_SIZE ) - MIN_CU_LOG2 + 1;
  int numPosIdx = MAX_CU_SIZE >> MIN_CU_LOG2;

  for( int xIdx = 0; xIdx < numPosIdx; xIdx++ )
  {
    for( int yIdx = 0; yIdx < numPosIdx; yIdx++ )
    {
      for( int wIdx = 0; wIdx < numSizeIdx; wIdx++ )
      {
        delete[] m_saveLoadSbt[xIdx][yIdx][wIdx];
      }
      delete[] m_saveLoadSbt[xIdx][yIdx];
    }
    delete[] m_saveLoadSbt[xIdx];
  }
  delete[] m_saveLoadSbt;
}

uint16_t SaveLoadEncInfoSbt::findBestSbt( const UnitArea& area, const uint32_t curPuSse )
{
  unsigned idx1, idx2, idx3, idx4;
  getAreaIdx( area.Y(), *m_sliceSbt->getPPS()->pcv, idx1, idx2, idx3, idx4 );
  SaveLoadStructSbt* pSbtSave = &m_saveLoadSbt[idx1][idx2][idx3 - MIN_CU_LOG2][idx4 - MIN_CU_LOG2];

  for( int i = 0; i < pSbtSave->numPuInfoStored; i++ )
  {
    if( curPuSse == pSbtSave->puSse[i] )
    {
      return pSbtSave->puSbt[i] + ( pSbtSave->puTrs[i] << 8 );
    }
  }

  return MAX_UCHAR + ( MAX_UCHAR << 8 );
}

bool SaveLoadEncInfoSbt::saveBestSbt( const UnitArea& area, const uint32_t curPuSse, const uint8_t curPuSbt, const uint8_t curPuTrs )
{
  unsigned idx1, idx2, idx3, idx4;
  getAreaIdx( area.Y(), *m_sliceSbt->getPPS()->pcv, idx1, idx2, idx3, idx4 );
  SaveLoadStructSbt* pSbtSave = &m_saveLoadSbt[idx1][idx2][idx3 - MIN_CU_LOG2][idx4 - MIN_CU_LOG2];

  if( pSbtSave->numPuInfoStored == SBT_NUM_SL )
  {
    return false;
  }

  pSbtSave->puSse[pSbtSave->numPuInfoStored] = curPuSse;
  pSbtSave->puSbt[pSbtSave->numPuInfoStored] = curPuSbt;
  pSbtSave->puTrs[pSbtSave->numPuInfoStored] = curPuTrs;
  pSbtSave->numPuInfoStored++;
  return true;
}

#if ENABLE_SPLIT_PARALLELISM
void SaveLoadEncInfoSbt::copyState(const SaveLoadEncInfoSbt &other)
{
  m_sliceSbt = other.m_sliceSbt;
}
#endif

void SaveLoadEncInfoSbt::resetSaveloadSbt( int maxSbtSize )
{
  int numSizeIdx = gp_sizeIdxInfo->idxFrom( maxSbtSize ) - MIN_CU_LOG2 + 1;
  int numPosIdx = MAX_CU_SIZE >> MIN_CU_LOG2;

  for( int xIdx = 0; xIdx < numPosIdx; xIdx++ )
  {
    for( int yIdx = 0; yIdx < numPosIdx; yIdx++ )
    {
      for( int wIdx = 0; wIdx < numSizeIdx; wIdx++ )
      {
        memset( m_saveLoadSbt[xIdx][yIdx][wIdx], 0, numSizeIdx * sizeof( SaveLoadStructSbt ) );
      }
    }
  }
}

bool CacheBlkInfoCtrl::getInter(const UnitArea& area)
{
  unsigned idx1, idx2, idx3, idx4;
  getAreaIdx(area.Y(), *m_slice_chblk->getPPS()->pcv, idx1, idx2, idx3, idx4);

  return m_codedCUInfo[idx1][idx2][idx3][idx4]->isInter;
}
void CacheBlkInfoCtrl::setBcwIdx(const UnitArea& area, uint8_t gBiIdx)
{
  unsigned idx1, idx2, idx3, idx4;
  getAreaIdx(area.Y(), *m_slice_chblk->getPPS()->pcv, idx1, idx2, idx3, idx4);

  m_codedCUInfo[idx1][idx2][idx3][idx4]->BcwIdx = gBiIdx;
}
uint8_t CacheBlkInfoCtrl::getBcwIdx(const UnitArea& area)
{
  unsigned idx1, idx2, idx3, idx4;
  getAreaIdx(area.Y(), *m_slice_chblk->getPPS()->pcv, idx1, idx2, idx3, idx4);

  return m_codedCUInfo[idx1][idx2][idx3][idx4]->BcwIdx;
}

#if REUSE_CU_RESULTS
static bool isTheSameNbHood( const CodingUnit &cu, const CodingStructure& cs, const Partitioner &partitioner
                            , const PredictionUnit &pu, int picW, int picH
                           )
{
  if( cu.chType != partitioner.chType )
  {
    return false;
  }

  const PartitioningStack &ps = partitioner.getPartStack();

  int i = 1;

  for( ; i < ps.size(); i++ )
  {
    if( ps[i].split != CU::getSplitAtDepth( cu, i - 1 ) )
    {
      break;
    }
  }

  const UnitArea &cmnAnc = ps[i - 1].parts[ps[i - 1].idx];
  const UnitArea cuArea  = CS::getArea( cs, cu, partitioner.chType );
//#endif

  for( int i = 0; i < cmnAnc.blocks.size(); i++ )
  {
    if( i < cuArea.blocks.size() && cuArea.blocks[i].valid() && cuArea.blocks[i].pos() != cmnAnc.blocks[i].pos() )
    {
      return false;
    }
  }

  return true;
}

void BestEncInfoCache::create( const ChromaFormat chFmt )
{
  const unsigned numPos = MAX_CU_SIZE >> MIN_CU_LOG2;

  m_numWidths  = gp_sizeIdxInfo->numWidths();
  m_numHeights = gp_sizeIdxInfo->numHeights();

  for( unsigned x = 0; x < numPos; x++ )
  {
    for( unsigned y = 0; y < numPos; y++ )
    {
      m_bestEncInfo[x][y] = new BestEncodingInfo**[m_numWidths];

      for( int wIdx = 0; wIdx < gp_sizeIdxInfo->numWidths(); wIdx++ )
      {
        if( gp_sizeIdxInfo->isCuSize( gp_sizeIdxInfo->sizeFrom( wIdx ) ) && x + ( gp_sizeIdxInfo->sizeFrom( wIdx ) >> MIN_CU_LOG2 ) <= ( MAX_CU_SIZE >> MIN_CU_LOG2 ) )
        {
          m_bestEncInfo[x][y][wIdx] = new BestEncodingInfo*[gp_sizeIdxInfo->numHeights()];

          for( int hIdx = 0; hIdx < gp_sizeIdxInfo->numHeights(); hIdx++ )
          {
            if( gp_sizeIdxInfo->isCuSize( gp_sizeIdxInfo->sizeFrom( hIdx ) ) && y + ( gp_sizeIdxInfo->sizeFrom( hIdx ) >> MIN_CU_LOG2 ) <= ( MAX_CU_SIZE >> MIN_CU_LOG2 ) )
            {
              m_bestEncInfo[x][y][wIdx][hIdx] = new BestEncodingInfo;

              int w = gp_sizeIdxInfo->sizeFrom( wIdx );
              int h = gp_sizeIdxInfo->sizeFrom( hIdx );

              const UnitArea area( chFmt, Area( 0, 0, w, h ) );

              new ( &m_bestEncInfo[x][y][wIdx][hIdx]->cu ) CodingUnit    ( area );
              new ( &m_bestEncInfo[x][y][wIdx][hIdx]->pu ) PredictionUnit( area );
#if REUSE_CU_RESULTS_WITH_MULTIPLE_TUS
              m_bestEncInfo[x][y][wIdx][hIdx]->numTus = 0;
              for( int i = 0; i < MAX_NUM_TUS; i++ )
              {
                new ( &m_bestEncInfo[x][y][wIdx][hIdx]->tus[i] ) TransformUnit( area );
              }
#else
              new ( &m_bestEncInfo[x][y][wIdx][hIdx]->tu ) TransformUnit( area );
#endif

              m_bestEncInfo[x][y][wIdx][hIdx]->poc      = -1;
              m_bestEncInfo[x][y][wIdx][hIdx]->testMode = EncTestMode();
            }
            else
            {
              m_bestEncInfo[x][y][wIdx][hIdx] = nullptr;
            }
          }
        }
        else
        {
          m_bestEncInfo[x][y][wIdx] = nullptr;
        }
      }
    }
  }
}

void BestEncInfoCache::destroy()
{
  const unsigned numPos = MAX_CU_SIZE >> MIN_CU_LOG2;

  for( unsigned x = 0; x < numPos; x++ )
  {
    for( unsigned y = 0; y < numPos; y++ )
    {
      for( int wIdx = 0; wIdx < gp_sizeIdxInfo->numWidths(); wIdx++ )
      {
        if( m_bestEncInfo[x][y][wIdx] )
        {
          for( int hIdx = 0; hIdx < gp_sizeIdxInfo->numHeights(); hIdx++ )
          {
            if( m_bestEncInfo[x][y][wIdx][hIdx] )
            {
              delete m_bestEncInfo[x][y][wIdx][hIdx];
            }
          }

          delete[] m_bestEncInfo[x][y][wIdx];
        }
      }

      delete[] m_bestEncInfo[x][y];
    }
  }

  delete[] m_pCoeff;
  delete[] m_pPcmBuf;

  if (m_runType != nullptr)
  {
    delete[] m_runType;
    m_runType = nullptr;
  }
}

void BestEncInfoCache::init( const Slice &slice )
{
  bool isInitialized = m_slice_bencinf;

  m_slice_bencinf = &slice;

  if( isInitialized ) return;

  const unsigned numPos = MAX_CU_SIZE >> MIN_CU_LOG2;

  m_numWidths  = gp_sizeIdxInfo->numWidths();
  m_numHeights = gp_sizeIdxInfo->numHeights();

  size_t numCoeff = 0;

  for( unsigned x = 0; x < numPos; x++ )
  {
    for( unsigned y = 0; y < numPos; y++ )
    {
      for( int wIdx = 0; wIdx < gp_sizeIdxInfo->numWidths(); wIdx++ )
      {
        if( m_bestEncInfo[x][y][wIdx] ) for( int hIdx = 0; hIdx < gp_sizeIdxInfo->numHeights(); hIdx++ )
        {
          if( m_bestEncInfo[x][y][wIdx][hIdx] )
          {
            for( const CompArea& blk : m_bestEncInfo[x][y][wIdx][hIdx]->cu.blocks )
            {
              numCoeff += blk.area();
            }
          }
        }
      }
    }
  }

#if REUSE_CU_RESULTS_WITH_MULTIPLE_TUS
  m_pCoeff  = new TCoeff[numCoeff*MAX_NUM_TUS];
  m_pPcmBuf = new Pel   [numCoeff*MAX_NUM_TUS];
  if (slice.getSPS()->getPLTMode())
  {
    m_runType   = new bool[numCoeff*MAX_NUM_TUS];
  }
#else
  m_pCoeff  = new TCoeff[numCoeff];
  m_pPcmBuf = new Pel   [numCoeff];
  if (slice.getSPS()->getPLTMode())
  {
    m_runType   = new bool[numCoeff];
  }
#endif

  TCoeff *coeffPtr = m_pCoeff;
  Pel    *pcmPtr   = m_pPcmBuf;
  bool   *runTypePtr   = m_runType;
  m_dummyCS.pcv = m_slice_bencinf->getPPS()->pcv;

  for( unsigned x = 0; x < numPos; x++ )
  {
    for( unsigned y = 0; y < numPos; y++ )
    {
      for( int wIdx = 0; wIdx < gp_sizeIdxInfo->numWidths(); wIdx++ )
      {
        if( m_bestEncInfo[x][y][wIdx] ) for( int hIdx = 0; hIdx < gp_sizeIdxInfo->numHeights(); hIdx++ )
        {
          if( m_bestEncInfo[x][y][wIdx][hIdx] )
          {
            TCoeff *coeff[MAX_NUM_TBLOCKS] = { 0, };
            Pel    *pcmbf[MAX_NUM_TBLOCKS] = { 0, };
            bool   *runType[MAX_NUM_TBLOCKS - 1] = { 0, };

#if REUSE_CU_RESULTS_WITH_MULTIPLE_TUS
            for( int i = 0; i < MAX_NUM_TUS; i++ )
            {
              TransformUnit &tu = m_bestEncInfo[x][y][wIdx][hIdx]->tus[i];
              const UnitArea &area = tu;

              for( int i = 0; i < area.blocks.size(); i++ )
              {
                coeff[i] = coeffPtr; coeffPtr += area.blocks[i].area();
                pcmbf[i] = pcmPtr;   pcmPtr += area.blocks[i].area();
                if (i < 2)
                {
                  runType[i]   = runTypePtr;   runTypePtr   += area.blocks[i].area();
                }
              }

              tu.cs = &m_dummyCS;
              tu.init(coeff, pcmbf, runType);
            }
#else
            const UnitArea &area = m_bestEncInfo[x][y][wIdx][hIdx]->tu;

            for( int i = 0; i < area.blocks.size(); i++ )
            {
              coeff[i] = coeffPtr; coeffPtr += area.blocks[i].area();
              pcmbf[i] =   pcmPtr;   pcmPtr += area.blocks[i].area();
              runType[i] = runTypePtr;     runTypePtr += area.blocks[i].area();
              runLength[i] = runLengthPtr; runLengthPtr += area.blocks[i].area();
            }

            m_bestEncInfo[x][y][wIdx][hIdx]->tu.cs = &m_dummyCS;
            m_bestEncInfo[x][y][wIdx][hIdx]->tu.init(coeff, pcmbf, runLength, runType);
#endif
          }
        }
      }
    }
  }
#if ENABLE_SPLIT_PARALLELISM

  m_currTemporalId = 0;
#endif
}

bool BestEncInfoCache::setFromCs( const CodingStructure& cs, const Partitioner& partitioner )
{
#if REUSE_CU_RESULTS_WITH_MULTIPLE_TUS
  if( cs.cus.size() != 1 || cs.pus.size() != 1 )
#else
  if( cs.cus.size() != 1 || cs.tus.size() != 1 || cs.pus.size() != 1 )
#endif
  {
    return false;
  }

  unsigned idx1, idx2, idx3, idx4;
  getAreaIdx( cs.area.Y(), *m_slice_bencinf->getPPS()->pcv, idx1, idx2, idx3, idx4 );

  BestEncodingInfo& encInfo = *m_bestEncInfo[idx1][idx2][idx3][idx4];

  encInfo.poc            =  cs.picture->poc;
  encInfo.cu.repositionTo( *cs.cus.front() );
  encInfo.pu.repositionTo( *cs.pus.front() );
#if !REUSE_CU_RESULTS_WITH_MULTIPLE_TUS
  encInfo.tu.repositionTo( *cs.tus.front() );
#endif
  encInfo.cu             = *cs.cus.front();
  encInfo.pu             = *cs.pus.front();
#if REUSE_CU_RESULTS_WITH_MULTIPLE_TUS
  int tuIdx = 0;
  for( auto tu : cs.tus )
  {
    encInfo.tus[tuIdx].repositionTo( *tu );
    encInfo.tus[tuIdx].resizeTo( *tu );
    for( auto &blk : tu->blocks )
    {
      if( blk.valid() )
        encInfo.tus[tuIdx].copyComponentFrom( *tu, blk.compID );
    }
    tuIdx++;
  }
  CHECKD( cs.tus.size() > MAX_NUM_TUS, "Exceeding tus array boundaries" );
  encInfo.numTus = cs.tus.size();
#else
  for( auto &blk : cs.tus.front()->blocks )
  {
    if( blk.valid() ) encInfo.tu.copyComponentFrom( *cs.tus.front(), blk.compID );
  }
#endif
  encInfo.testMode       = getCSEncMode( cs );

  return true;
}

bool BestEncInfoCache::isValid( const CodingStructure& cs, const Partitioner& partitioner, int qp )
{
  if( partitioner.treeType == TREE_C )
  {
    return false; //if save & load is allowed for chroma CUs, we should check whether luma info (pred, recon, etc) is the same, which is quite complex
  }
  unsigned idx1, idx2, idx3, idx4;
  getAreaIdx( cs.area.Y(), *m_slice_bencinf->getPPS()->pcv, idx1, idx2, idx3, idx4 );

  BestEncodingInfo& encInfo = *m_bestEncInfo[idx1][idx2][idx3][idx4];

  if( encInfo.cu.treeType != partitioner.treeType || encInfo.cu.modeType != partitioner.modeType )
  {
    return false;
  }
  if( encInfo.cu.qp != qp )
    return false;
  if( cs.picture->poc != encInfo.poc || CS::getArea( cs, cs.area, partitioner.chType ) != CS::getArea( cs, encInfo.cu, partitioner.chType ) || !isTheSameNbHood( encInfo.cu, cs, partitioner
    , encInfo.pu, (cs.picture->Y().width), (cs.picture->Y().height)
)
    || CU::isIBC(encInfo.cu)
    || partitioner.currQgEnable() || cs.currQP[partitioner.chType] != encInfo.cu.qp
    )
  {
    return false;
  }
  else
  {
    return true;
  }
}

bool BestEncInfoCache::setCsFrom( CodingStructure& cs, EncTestMode& testMode, const Partitioner& partitioner ) const
{
  unsigned idx1, idx2, idx3, idx4;
  getAreaIdx( cs.area.Y(), *m_slice_bencinf->getPPS()->pcv, idx1, idx2, idx3, idx4 );

  BestEncodingInfo& encInfo = *m_bestEncInfo[idx1][idx2][idx3][idx4];

  if( cs.picture->poc != encInfo.poc || CS::getArea( cs, cs.area, partitioner.chType ) != CS::getArea( cs, encInfo.cu, partitioner.chType ) || !isTheSameNbHood( encInfo.cu, cs, partitioner
    , encInfo.pu, (cs.picture->Y().width), (cs.picture->Y().height)
    )
    || partitioner.currQgEnable() || cs.currQP[partitioner.chType] != encInfo.cu.qp
    )
  {
    return false;
  }

  CodingUnit     &cu = cs.addCU( CS::getArea( cs, cs.area, partitioner.chType ), partitioner.chType );
  PredictionUnit &pu = cs.addPU( CS::getArea( cs, cs.area, partitioner.chType ), partitioner.chType );
#if !REUSE_CU_RESULTS_WITH_MULTIPLE_TUS
  TransformUnit  &tu = cs.addTU( CS::getArea( cs, cs.area, partitioner.chType ), partitioner.chType );
#endif

  cu          .repositionTo( encInfo.cu );
  pu          .repositionTo( encInfo.pu );
#if !REUSE_CU_RESULTS_WITH_MULTIPLE_TUS
  tu          .repositionTo( encInfo.tu );
#endif

  cu          = encInfo.cu;
  pu          = encInfo.pu;
#if REUSE_CU_RESULTS_WITH_MULTIPLE_TUS
  CHECKD( !( encInfo.numTus > 0 ), "Empty tus array" );
  for( int i = 0; i < encInfo.numTus; i++ )
  {
    TransformUnit  &tu = cs.addTU( encInfo.tus[i], partitioner.chType );

    for( auto &blk : tu.blocks )
    {
      if( blk.valid() ) tu.copyComponentFrom( encInfo.tus[i], blk.compID );
    }
  }
#else
  for( auto &blk : tu.blocks )
  {
    if( blk.valid() ) tu.copyComponentFrom( encInfo.tu, blk.compID );
  }
#endif

  testMode    = encInfo.testMode;

  return true;
}

#if ENABLE_SPLIT_PARALLELISM
void BestEncInfoCache::copyState(const BestEncInfoCache &other, const UnitArea &area)
{
  m_slice_bencinf  = other.m_slice_bencinf;
  m_currTemporalId = other.m_currTemporalId;

  if( m_slice_bencinf->isIntra() ) return;

  const int cuSizeMask = m_slice_bencinf->getSPS()->getMaxCUWidth() - 1;

  const int minPosX = ( area.lx() & cuSizeMask ) >> MIN_CU_LOG2;
  const int minPosY = ( area.ly() & cuSizeMask ) >> MIN_CU_LOG2;
  const int maxPosX = ( area.Y().bottomRight().x & cuSizeMask ) >> MIN_CU_LOG2;
  const int maxPosY = ( area.Y().bottomRight().y & cuSizeMask ) >> MIN_CU_LOG2;

  for( unsigned x = minPosX; x <= maxPosX; x++ )
  {
    for( unsigned y = minPosY; y <= maxPosY; y++ )
    {
      for( int wIdx = 0; wIdx < gp_sizeIdxInfo->numWidths(); wIdx++ )
      {
        const int width = gp_sizeIdxInfo->sizeFrom( wIdx );

        if( m_bestEncInfo[x][y][wIdx] && width <= area.lwidth() && x + ( width >> MIN_CU_LOG2 ) <= ( maxPosX + 1 ) )
        {
          for( int hIdx = 0; hIdx < gp_sizeIdxInfo->numHeights(); hIdx++ )
          {
            const int height = gp_sizeIdxInfo->sizeFrom( hIdx );

            if( gp_sizeIdxInfo->isCuSize( height ) && height <= area.lheight() && y + ( height >> MIN_CU_LOG2 ) <= ( maxPosY + 1 ) )
            {
              if( other.m_bestEncInfo[x][y][wIdx][hIdx]->temporalId > m_bestEncInfo[x][y][wIdx][hIdx]->temporalId )
              {
                m_bestEncInfo[x][y][wIdx][hIdx]->cu       = other.m_bestEncInfo[x][y][wIdx][hIdx]->cu;
                m_bestEncInfo[x][y][wIdx][hIdx]->pu       = other.m_bestEncInfo[x][y][wIdx][hIdx]->pu;
                m_bestEncInfo[x][y][wIdx][hIdx]->numTus   = other.m_bestEncInfo[x][y][wIdx][hIdx]->numTus;
                m_bestEncInfo[x][y][wIdx][hIdx]->poc      = other.m_bestEncInfo[x][y][wIdx][hIdx]->poc;
                m_bestEncInfo[x][y][wIdx][hIdx]->testMode = other.m_bestEncInfo[x][y][wIdx][hIdx]->testMode;

                for( int i = 0; i < m_bestEncInfo[x][y][wIdx][hIdx]->numTus; i++ )
                  m_bestEncInfo[x][y][wIdx][hIdx]->tus[i] = other.m_bestEncInfo[x][y][wIdx][hIdx]->tus[i];
              }
            }
            else if( y + ( height >> MIN_CU_LOG2 ) > maxPosY + 1 )
            {
              break;;
            }
          }
        }
        else if( x + ( width >> MIN_CU_LOG2 ) > maxPosX + 1 )
        {
          break;
        }
      }
    }
  }
}

void BestEncInfoCache::touch(const UnitArea &area)
{
  unsigned idx1, idx2, idx3, idx4;
  getAreaIdx(area.Y(), *m_slice_bencinf->getPPS()->pcv, idx1, idx2, idx3, idx4);
  BestEncodingInfo &encInfo = *m_bestEncInfo[idx1][idx2][idx3][idx4];

  encInfo.temporalId = m_currTemporalId;
}

#endif

#endif

static bool interHadActive( const ComprCUCtx& ctx )
{
  return ctx.interHad != 0;
}

//////////////////////////////////////////////////////////////////////////
// EncModeCtrlQTBT
//////////////////////////////////////////////////////////////////////////

void EncModeCtrlMTnoRQT::create( const EncCfg& cfg )
{
  CacheBlkInfoCtrl::create();
#if REUSE_CU_RESULTS
  BestEncInfoCache::create( cfg.getChromaFormatIdc() );
#endif
  SaveLoadEncInfoSbt::create();
}

void EncModeCtrlMTnoRQT::destroy()
{
  CacheBlkInfoCtrl::destroy();
#if REUSE_CU_RESULTS
  BestEncInfoCache::destroy();
#endif
  SaveLoadEncInfoSbt::destroy();
}

void EncModeCtrlMTnoRQT::initCTUEncoding( const Slice &slice )
{
  CacheBlkInfoCtrl::init( slice );
#if REUSE_CU_RESULTS
  BestEncInfoCache::init( slice );
#endif
  SaveLoadEncInfoSbt::init( slice );

  CHECK( !m_ComprCUCtxList.empty(), "Mode list is not empty at the beginning of a CTU" );

  m_slice             = &slice;
#if ENABLE_SPLIT_PARALLELISM
  m_runNextInParallel      = false;
#endif

  if( m_pcEncCfg->getUseE0023FastEnc() )
  {
    if (m_pcEncCfg->getUseCompositeRef())
      m_skipThreshold = ( ( slice.getMinPictureDistance() <= PICTURE_DISTANCE_TH * 2 ) ? FAST_SKIP_DEPTH : SKIP_DEPTH );
    else
      m_skipThreshold = ((slice.getMinPictureDistance() <= PICTURE_DISTANCE_TH) ? FAST_SKIP_DEPTH : SKIP_DEPTH);

  }
  else
  {
    m_skipThreshold = SKIP_DEPTH;
  }
}

void EncModeCtrlMTnoRQT::initCULevel( Partitioner &partitioner, const CodingStructure& cs )
{
  // Min/max depth
  unsigned minDepth = 0;
  unsigned maxDepth = floorLog2(cs.sps->getCTUSize()) - floorLog2(cs.sps->getMinQTSize( m_slice->getSliceType(), partitioner.chType ));
  if( m_pcEncCfg->getUseFastLCTU() )
  {
    if( auto adPartitioner = dynamic_cast<AdaptiveDepthPartitioner*>( &partitioner ) )
    {
      // LARGE CTU
      adPartitioner->setMaxMinDepth( minDepth, maxDepth, cs );
    }
  }

  m_ComprCUCtxList.push_back( ComprCUCtx( cs, minDepth, maxDepth, NUM_EXTRA_FEATURES ) );

#if ENABLE_SPLIT_PARALLELISM
  if( m_runNextInParallel )
  {
    for( auto &level : m_ComprCUCtxList )
    {
      CHECK( level.isLevelSplitParallel, "Tring to parallelize a level within parallel execution!" );
    }
    CHECK( cs.picture->scheduler.getSplitJobId() == 0, "Trying to run a parallel level although jobId is 0!" );
    m_runNextInParallel                          = false;
    m_ComprCUCtxList.back().isLevelSplitParallel = true;
  }

#endif
  const CodingUnit* cuLeft  = cs.getCU( cs.area.blocks[partitioner.chType].pos().offset( -1, 0 ), partitioner.chType );
  const CodingUnit* cuAbove = cs.getCU( cs.area.blocks[partitioner.chType].pos().offset( 0, -1 ), partitioner.chType );


  const bool qtBeforeBt = ((cuLeft && cuAbove && cuLeft->qtDepth > partitioner.currQtDepth && cuAbove->qtDepth > partitioner.currQtDepth)
	  || (cuLeft && !cuAbove && cuLeft->qtDepth > partitioner.currQtDepth)
	  || (!cuLeft && cuAbove && cuAbove->qtDepth > partitioner.currQtDepth)
	  || (!cuAbove && !cuLeft && cs.area.lwidth() >= (32 << cs.slice->getDepth())))
	  && (cs.area.lwidth() > (cs.pcv->getMinQtSize(*cs.slice, partitioner.chType) << 1));


  // set features
  ComprCUCtx &cuECtx  = m_ComprCUCtxList.back();
  cuECtx.set( BEST_NON_SPLIT_COST,  MAX_DOUBLE );
  cuECtx.set( BEST_VERT_SPLIT_COST, MAX_DOUBLE );
  cuECtx.set( BEST_HORZ_SPLIT_COST, MAX_DOUBLE );
  cuECtx.set( BEST_TRIH_SPLIT_COST, MAX_DOUBLE );
  cuECtx.set( BEST_TRIV_SPLIT_COST, MAX_DOUBLE );
  cuECtx.set( DO_TRIH_SPLIT,        1 );
  cuECtx.set( DO_TRIV_SPLIT,        1 );
  cuECtx.set( BEST_IMV_COST,        MAX_DOUBLE * .5 );
  cuECtx.set( BEST_NO_IMV_COST,     MAX_DOUBLE * .5 );
  cuECtx.set( QT_BEFORE_BT,         qtBeforeBt );
  cuECtx.set( DID_QUAD_SPLIT,       false );
  cuECtx.set( IS_BEST_NOSPLIT_SKIP, false );
  cuECtx.set( MAX_QT_SUB_DEPTH,     0 );
#if  FEATURE_TEST
  cuECtx.set(BEST_QT_COST, MAX_DOUBLE);
  cuECtx.set(NO_SPLIT_FLAG, 2);
  cuECtx.set(QT_FLAG, 2);
  cuECtx.set(HOR_FLAG, 2);
#endif
#if DISABLE_RF_IF_EMPTY_CU_WHEN_FULL
  cuECtx.set(EMPTY_CU_WHEN_FULL, false);
#endif
#if FEATURE_TEST
  cuECtx.set(IS_NON_SPLIT_INTRA, false);
  cuECtx.set(IS_NON_SPLIT_INTER, false);
  cuECtx.set(IS_NON_SPLIT_MERGE, false);
  cuECtx.set(IS_NON_SPLIT_GEO, false);
#endif
  // QP
  int baseQP = cs.baseQP;
  if (!partitioner.isSepTree(cs) || isLuma(partitioner.chType))
  {
    if (m_pcEncCfg->getUseAdaptiveQP())
    {
      baseQP = Clip3(-cs.sps->getQpBDOffset(CHANNEL_TYPE_LUMA), MAX_QP, baseQP + xComputeDQP(cs, partitioner));
    }
#if ENABLE_QPA_SUB_CTU
    else if (m_pcEncCfg->getUsePerceptQPA() && !m_pcEncCfg->getUseRateCtrl() && cs.pps->getUseDQP() && cs.slice->getCuQpDeltaSubdiv() > 0)
    {
      const PreCalcValues &pcv = *cs.pcv;

      if ((partitioner.currArea().lwidth() < pcv.maxCUWidth) && (partitioner.currArea().lheight() < pcv.maxCUHeight) && cs.picture)
      {
        const Position    &pos = partitioner.currQgPos;
        const unsigned mtsLog2 = (unsigned)floorLog2(std::min (cs.sps->getMaxTbSize(), pcv.maxCUWidth));
        const unsigned  stride = pcv.maxCUWidth >> mtsLog2;

        baseQP = cs.picture->m_subCtuQP[((pos.x & pcv.maxCUWidthMask) >> mtsLog2) + stride * ((pos.y & pcv.maxCUHeightMask) >> mtsLog2)];
      }
    }
#endif
#if SHARP_LUMA_DELTA_QP
    if (m_pcEncCfg->getLumaLevelToDeltaQPMapping().isEnabled())
    {
      if (partitioner.currQgEnable())
      {
        m_lumaQPOffset = calculateLumaDQP (cs.getOrgBuf (clipArea (cs.area.Y(), cs.picture->Y())));
      }
      baseQP = Clip3 (-cs.sps->getQpBDOffset (CHANNEL_TYPE_LUMA), MAX_QP, baseQP - m_lumaQPOffset);
    }
#endif
  }
  int minQP = baseQP;
  int maxQP = baseQP;

  xGetMinMaxQP( minQP, maxQP, cs, partitioner, baseQP, *cs.sps, *cs.pps, CU_QUAD_SPLIT );
  bool checkIbc = true;
  if (partitioner.chType == CHANNEL_TYPE_CHROMA)
  {
    checkIbc = false;
  }
  // Add coding modes here
  // NOTE: Working back to front, as a stack, which is more efficient with the container
  // NOTE: First added modes will be processed at the end.

  //////////////////////////////////////////////////////////////////////////
  // Add unit split modes


  if( !cuECtx.get<bool>( QT_BEFORE_BT ) )
  {
    for( int qp = maxQP; qp >= minQP; qp-- )
    {
      m_ComprCUCtxList.back().testModes.push_back( { ETM_SPLIT_QT, ETO_STANDARD, qp } );
    }
  }

  if( partitioner.canSplit( CU_TRIV_SPLIT, cs ) )
  {
    // add split modes
    for( int qp = maxQP; qp >= minQP; qp-- )
    {
      m_ComprCUCtxList.back().testModes.push_back( { ETM_SPLIT_TT_V, ETO_STANDARD, qp } );
    }
  }

  if( partitioner.canSplit( CU_TRIH_SPLIT, cs ) )
  {
    // add split modes
    for( int qp = maxQP; qp >= minQP; qp-- )
    {
      m_ComprCUCtxList.back().testModes.push_back( { ETM_SPLIT_TT_H, ETO_STANDARD, qp } );
    }
  }



  int minQPq = minQP;
  int maxQPq = maxQP;
  xGetMinMaxQP( minQP, maxQP, cs, partitioner, baseQP, *cs.sps, *cs.pps, CU_BT_SPLIT );

  if( partitioner.canSplit( CU_VERT_SPLIT, cs ) )
  {
    // add split modes
    for( int qp = maxQP; qp >= minQP; qp-- )
    {
      m_ComprCUCtxList.back().testModes.push_back( { ETM_SPLIT_BT_V, ETO_STANDARD, qp } );
    }
    m_ComprCUCtxList.back().set( DID_VERT_SPLIT, true );
  }
  else

  {
    m_ComprCUCtxList.back().set( DID_VERT_SPLIT, false );
  }

  if( partitioner.canSplit( CU_HORZ_SPLIT, cs ) )
  {
    // add split modes
    for( int qp = maxQP; qp >= minQP; qp-- )
    {
      m_ComprCUCtxList.back().testModes.push_back( { ETM_SPLIT_BT_H, ETO_STANDARD, qp } );
    }
    m_ComprCUCtxList.back().set( DID_HORZ_SPLIT, true );
  }
  else
  {
    m_ComprCUCtxList.back().set( DID_HORZ_SPLIT, false );
  }


  if( cuECtx.get<bool>( QT_BEFORE_BT ) )
  {
    for( int qp = maxQPq; qp >= minQPq; qp-- )
    {
      m_ComprCUCtxList.back().testModes.push_back( { ETM_SPLIT_QT, ETO_STANDARD, qp } );
    }
  }


  m_ComprCUCtxList.back().testModes.push_back( { ETM_POST_DONT_SPLIT } );

  xGetMinMaxQP( minQP, maxQP, cs, partitioner, baseQP, *cs.sps, *cs.pps, CU_DONT_SPLIT );

  int  lowestQP = minQP;

  //////////////////////////////////////////////////////////////////////////
  // Add unit coding modes: Intra, InterME, InterMerge ...
  bool tryIntraRdo = true;
  bool tryInterRdo = true;
  bool tryIBCRdo   = true;
  if( partitioner.isConsIntra() )
  {
    tryInterRdo = false;
  }
  else if( partitioner.isConsInter() )
  {
    tryIntraRdo = tryIBCRdo = false;
  }
  checkIbc &= tryIBCRdo;

  for( int qpLoop = maxQP; qpLoop >= minQP; qpLoop-- )
  {
    const int  qp       = std::max( qpLoop, lowestQP );
#if REUSE_CU_RESULTS
    const bool isReusingCu = isValid( cs, partitioner, qp );
#if DISABLE_CU_CACHING
	cuECtx.set(IS_REUSING_CU, false);
#else
    cuECtx.set( IS_REUSING_CU, isReusingCu );
#endif
    if( isReusingCu )
    {
      m_ComprCUCtxList.back().testModes.push_back( {ETM_RECO_CACHED, ETO_STANDARD, qp} );
    }
#endif
    // add intra modes
    if( tryIntraRdo )
    {
#if JVET_Q0504_PLT_NON444
    if (cs.slice->getSPS()->getPLTMode() && (partitioner.treeType != TREE_D || cs.slice->isIRAP() || (cs.area.lwidth() == 4 && cs.area.lheight() == 4)) && getPltEnc())
#else
    if (cs.slice->getSPS()->getPLTMode() && ( cs.slice->isIRAP() || (cs.area.lwidth() == 4 && cs.area.lheight() == 4) ) && getPltEnc() )
#endif
    {
      m_ComprCUCtxList.back().testModes.push_back({ ETM_PALETTE, ETO_STANDARD, qp });
    }
    m_ComprCUCtxList.back().testModes.push_back( { ETM_INTRA, ETO_STANDARD, qp } );
#if JVET_Q0504_PLT_NON444
    if (cs.slice->getSPS()->getPLTMode() && partitioner.treeType == TREE_D && !cs.slice->isIRAP() && !(cs.area.lwidth() == 4 && cs.area.lheight() == 4) && getPltEnc())
#else
    if (cs.slice->getSPS()->getPLTMode() && !cs.slice->isIRAP() && !(cs.area.lwidth() == 4 && cs.area.lheight() == 4) && getPltEnc() )
#endif
    {
      m_ComprCUCtxList.back().testModes.push_back({ ETM_PALETTE,  ETO_STANDARD, qp });
    }
    }
    // add ibc mode to intra path
    if (cs.sps->getIBCFlag() && checkIbc)
    {
      m_ComprCUCtxList.back().testModes.push_back({ ETM_IBC,         ETO_STANDARD,  qp });
      if (partitioner.chType == CHANNEL_TYPE_LUMA)
      {
        m_ComprCUCtxList.back().testModes.push_back({ ETM_IBC_MERGE,   ETO_STANDARD,  qp });
      }
    }
  }

  // add first pass modes
  if ( !m_slice->isIRAP() && !( cs.area.lwidth() == 4 && cs.area.lheight() == 4 ) && tryInterRdo )
  {
    for( int qpLoop = maxQP; qpLoop >= minQP; qpLoop-- )
    {
      const int  qp       = std::max( qpLoop, lowestQP );
      if (m_pcEncCfg->getIMV())
      {
        m_ComprCUCtxList.back().testModes.push_back({ ETM_INTER_ME,  EncTestModeOpts( 4 << ETO_IMV_SHIFT ), qp });
      }
      if( m_pcEncCfg->getIMV() || m_pcEncCfg->getUseAffineAmvr() )
      {
        int imv = m_pcEncCfg->getIMV4PelFast() ? 3 : 2;
        m_ComprCUCtxList.back().testModes.push_back( { ETM_INTER_ME, EncTestModeOpts( imv << ETO_IMV_SHIFT ), qp } );
        m_ComprCUCtxList.back().testModes.push_back( { ETM_INTER_ME, EncTestModeOpts( 1 << ETO_IMV_SHIFT ), qp } );
      }
      // add inter modes
      if( m_pcEncCfg->getUseEarlySkipDetection() )
      {
#if !JVET_Q0806
        if( cs.sps->getUseTriangle() && cs.slice->isInterB() )
        {
          m_ComprCUCtxList.back().testModes.push_back( { ETM_MERGE_TRIANGLE, ETO_STANDARD, qp } );
        }
#else
        if( cs.sps->getUseGeo() && cs.slice->isInterB() )
        {
          m_ComprCUCtxList.back().testModes.push_back( { ETM_MERGE_GEO, ETO_STANDARD, qp } );
        }
#endif
        m_ComprCUCtxList.back().testModes.push_back( { ETM_MERGE_SKIP,  ETO_STANDARD, qp } );
        if ( cs.sps->getUseAffine() || cs.sps->getSBTMVPEnabledFlag() )
        {
          m_ComprCUCtxList.back().testModes.push_back( { ETM_AFFINE,    ETO_STANDARD, qp } );
        }
        m_ComprCUCtxList.back().testModes.push_back( { ETM_INTER_ME,    ETO_STANDARD, qp } );
      }
      else
      {
        m_ComprCUCtxList.back().testModes.push_back( { ETM_INTER_ME,    ETO_STANDARD, qp } );
#if !JVET_Q0806
        if( cs.sps->getUseTriangle() && cs.slice->isInterB() )
        {
          m_ComprCUCtxList.back().testModes.push_back( { ETM_MERGE_TRIANGLE, ETO_STANDARD, qp } );
        }
#else
        if( cs.sps->getUseGeo() && cs.slice->isInterB() )
        {
          m_ComprCUCtxList.back().testModes.push_back( { ETM_MERGE_GEO, ETO_STANDARD, qp } );
        }
#endif
        m_ComprCUCtxList.back().testModes.push_back( { ETM_MERGE_SKIP,  ETO_STANDARD, qp } );
        if ( cs.sps->getUseAffine() || cs.sps->getSBTMVPEnabledFlag() )
        {
          m_ComprCUCtxList.back().testModes.push_back( { ETM_AFFINE,    ETO_STANDARD, qp } );
        }
      }
      if (m_pcEncCfg->getUseHashME())
      {
        int minSize = min(cs.area.lwidth(), cs.area.lheight());
        if (minSize < 128 && minSize >= 4)
        {
          m_ComprCUCtxList.back().testModes.push_back({ ETM_HASH_INTER, ETO_STANDARD, qp });
        }
      }
    }
  }

  // ensure to skip unprobable modes
#if FEATURE_TEST
  m_ComprCUCtxList.back().testModes.push_back({ IS_FIRST_MODE });
#endif
  if( !tryModeMaster( m_ComprCUCtxList.back().testModes.back(), cs, partitioner ) )
  {
    nextMode( cs, partitioner );
  }

  m_ComprCUCtxList.back().lastTestMode = EncTestMode();
}

void EncModeCtrlMTnoRQT::finishCULevel( Partitioner &partitioner )
{
  m_ComprCUCtxList.pop_back();
}


bool EncModeCtrlMTnoRQT::tryMode( const EncTestMode& encTestmode, const CodingStructure &cs, Partitioner& partitioner )
{
  ComprCUCtx& cuECtx = m_ComprCUCtxList.back();

#if FEATURE_TEST
  const CompArea& currArea = partitioner.currArea().Y();
  int ht = currArea.height, wd = currArea.width;

  int sizeIndex = 28;
  switch (wd)
  {
  case 128:
	  switch (ht)
	  {
	  case 128:
		  sizeIndex = 27;
		  break;
	  case 64:
		  sizeIndex = 26;
		  break;
	  }
	  break;
  case 64:
	  switch (ht)
	  {
	  case 128:
		  sizeIndex = 25;
		  break;
	  case 64:
		  sizeIndex = 24;
		  break;
	  case 32:
		  sizeIndex = 23;
		  break;
	  case 16:
		  sizeIndex = 20;
		  break;
	  case 8:
		  sizeIndex = 15;
		  break;
	  }
	  break;
  case 32:
	  switch (ht)
	  {
	  case 64:
		  sizeIndex = 22;
		  break;
	  case 32:
		  sizeIndex = 21;
		  break;
	  case 16:
		  sizeIndex = 18;
		  break;
	  case 8:
		  sizeIndex = 13;
		  break;
	  }
	  break;
  case 16:
	  switch (ht)
	  {
	  case 64:
		  sizeIndex = 19;
		  break;
	  case 32:
		  sizeIndex = 17;
		  break;
	  case 16:
		  sizeIndex = 16;
		  break;
	  case 8:
		  sizeIndex = 11;
		  break;
	  }
	  break;
  case 8:
	  switch (ht)
	  {
	  case 64:
		  sizeIndex = 14;
		  break;
	  case 32:
		  sizeIndex = 12;
		  break;
	  case 16:
		  sizeIndex = 10;
		  break;
	  case 8:
		  sizeIndex = 9;
		  break;
	  }
	  break;
  }

  if (encTestmode.type == ETM_POST_DONT_SPLIT)
  {
	  if (cs.slice->getSliceType() != I_SLICE)
	  {
		  if (sizeIndex == 28)
			  return false;

		  int qp = cs.baseQP;

		  const CompArea& currArea = partitioner.currArea().Y();
		  int ht = currArea.height,  wd = currArea.width;
		  if ((ht + partitioner.currArea().Y().lumaPos().y >= cs.slice->getPic()->lheight()) || (wd + partitioner.currArea().Y().lumaPos().x >= cs.slice->getPic()->lwidth()))
		  {
			  return false;
		  }
		  const CPelBuf orgPel = cs.getOrgBuf(partitioner.currArea().block(COMPONENT_Y)); //cs.getOrgBuf(cs.area.blocks[COMPONENT_Y]);
		  int halfW = max(wd / 2, 4), halfH = max(ht / 2, 4);
		  double sum = 0, squaredSum = 0, totalSum = 0, totalSquaredSum = 0;


		  for (int y = 0; y < halfH; y++)
		  {
			  for (int x = 0; x < halfW; x++)
			  {
				  const Pel* ptrOrg = orgPel.bufAt(x, y);
				  sum += (int)* ptrOrg;  squaredSum += ((int)* ptrOrg) * ((int)* ptrOrg);
			  }
		  }
		  double aveTopL = (double)sum / (double)(halfH * halfW);
		  double varTopL = ((double)(squaredSum) / (double)(halfH * halfW)) - (aveTopL * aveTopL);
		  totalSum += sum;  totalSquaredSum += squaredSum;
		  double _squaredSumPixTopL = squaredSum, _sumPixTopL = sum;

		  sum = 0, squaredSum = 0;
		  for (int y = 0; y < halfH; y++)
		  {
			  for (int x = halfW; x < wd; x++)
			  {
				  const Pel* ptrOrg = orgPel.bufAt(x, y);
				  sum += (int)* ptrOrg;  squaredSum += ((int)* ptrOrg) * ((int)* ptrOrg);
			  }
		  }
		  double aveTopR = (double)sum / (double)(halfH * halfW);
		  double varTopR = ((double)(squaredSum) / (double)(halfH * halfW)) - (aveTopR * aveTopR);
		  totalSum += sum;  totalSquaredSum += squaredSum;
		  double _squaredSumPixTopR = squaredSum, _sumPixTopR = sum;

		  sum = 0, squaredSum = 0;
		  for (int y = halfH; y < ht; y++)
		  {
			  for (int x = 0; x < halfW; x++)
			  {
				  const Pel* ptrOrg = orgPel.bufAt(x, y);
				  sum += (int)* ptrOrg;  squaredSum += ((int)* ptrOrg) * ((int)* ptrOrg);
			  }
		  }
		  double aveBotL = (double)sum / (double)(halfH * halfW);
		  double varBotL = ((double)(squaredSum) / (double)(halfH * halfW)) - (aveBotL * aveBotL);
		  totalSum += sum;  totalSquaredSum += squaredSum;
		  double _squaredSumPixBotL = squaredSum, _sumPixBotL= sum;

		  sum = 0, squaredSum = 0;
		  for (int y = halfH; y < ht; y++)
		  {
			  for (int x = halfW; x < wd; x++)
			  {
				  const Pel* ptrOrg = orgPel.bufAt(x, y);
				  sum += (int)* ptrOrg;  squaredSum += ((int)* ptrOrg) * ((int)* ptrOrg);
			  }
		  }
		  double aveBotR = (double)sum / (double)(halfH * halfW);
		  double varBotR = ((double)(squaredSum) / (double)(halfH * halfW)) - (aveBotR * aveBotR);
		  totalSum += sum;  totalSquaredSum += squaredSum;
		  double _squaredSumPixBotR = squaredSum, _sumPixBotR = sum;

		  double ave = ((double)totalSum / (double)(ht * wd));                       //redundant
		  double var = ((double)(totalSquaredSum) / (double)(ht * wd)) - (ave * ave);

		  double gradHor = 0.0, gradVer = 0.0;
		  double gradHorTopL = 0.0, gradVerTopL = 0.0;
		  double sobel = 0.0;
		  double sobelTopL = 0.0;
		  for (int y = 0; y < halfH - 1; y++)
		  {
			  for (int x = 0; x < halfW - 1; x++)
			  {
				  gradHorTopL += abs(orgPel.at(x + 1, y) - orgPel.at(x, y));
				  gradVerTopL += abs(orgPel.at(x, y + 1) - orgPel.at(x, y));
				  if (x >= 1 && y >= 1)
				  {
					  double x_sum = (orgPel.at(x - 1, y + 1) + 2 * orgPel.at(x, y + 1) + orgPel.at(x + 1, y + 1))
						  - (orgPel.at(x - 1, y - 1) + 2 * orgPel.at(x, y - 1) + orgPel.at(x + 1, y - 1));
					  double y_sum = (orgPel.at(x + 1, y - 1) + 2 * orgPel.at(x + 1, y) + orgPel.at(x + 1, y + 1))
						  - (orgPel.at(x - 1, y - 1) + 2 * orgPel.at(x - 1, y) + orgPel.at(x - 1, y + 1));
					  sobelTopL += ((x_sum) * (x_sum)) + ((y_sum) * (y_sum));
				  }
			  }
		  }
		  sobel += sobelTopL;
		  sobelTopL = sobelTopL / ((halfH - 2) * (halfW - 2));
		  gradHor += gradHorTopL;   gradVer += gradVerTopL;
		  gradHorTopL = gradHorTopL / ((halfH - 1) * (halfW - 1));
		  gradVerTopL = gradVerTopL / ((halfH - 1) * (halfW - 1));


		  double gradHorTopR = 0.0, gradVerTopR = 0.0;
		  double sobelTopR = 0.0;
		  for (int y = 0; y < halfH - 1; y++)
		  {
			  for (int x = halfW; x < wd - 1; x++)
			  {
				  gradHorTopR += abs(orgPel.at(x + 1, y) - orgPel.at(x, y));
				  gradVerTopR += abs(orgPel.at(x, y + 1) - orgPel.at(x, y));
				  if (x >= 1 && y >= 1)
				  {
					  double x_sum = (orgPel.at(x - 1, y + 1) + 2 * orgPel.at(x, y + 1) + orgPel.at(x + 1, y + 1))
						  - (orgPel.at(x - 1, y - 1) + 2 * orgPel.at(x, y - 1) + orgPel.at(x + 1, y - 1));
					  double y_sum = (orgPel.at(x + 1, y - 1) + 2 * orgPel.at(x + 1, y) + orgPel.at(x + 1, y + 1))
						  - (orgPel.at(x - 1, y - 1) + 2 * orgPel.at(x - 1, y) + orgPel.at(x - 1, y + 1));
					  sobelTopR += ((x_sum) * (x_sum)) + ((y_sum) * (y_sum));
				  }
			  }
		  }
		  sobel += sobelTopR;
		  sobelTopR = sobelTopR / ((halfH - 2) * (halfW - 2));
		  gradHor += gradHorTopR;   gradVer += gradVerTopR;
		  gradHorTopR = gradHorTopR / ((halfH - 1) * (halfW - 1));
		  gradVerTopR = gradVerTopR / ((halfH - 1) * (halfW - 1));


		  double gradHorBotL = 0.0, gradVerBotL = 0.0;
		  double sobelBotL = 0.0;
		  for (int y = halfH; y < ht - 1; y++)
		  {
			  for (int x = 0; x < halfW - 1; x++)
			  {
				  gradHorBotL += abs(orgPel.at(x + 1, y) - orgPel.at(x, y));
				  gradVerBotL += abs(orgPel.at(x, y + 1) - orgPel.at(x, y));
				  if (x >= 1 && y >= 1)
				  {
					  double x_sum = (orgPel.at(x - 1, y + 1) + 2 * orgPel.at(x, y + 1) + orgPel.at(x + 1, y + 1))
						  - (orgPel.at(x - 1, y - 1) + 2 * orgPel.at(x, y - 1) + orgPel.at(x + 1, y - 1));
					  double y_sum = (orgPel.at(x + 1, y - 1) + 2 * orgPel.at(x + 1, y) + orgPel.at(x + 1, y + 1))
						  - (orgPel.at(x - 1, y - 1) + 2 * orgPel.at(x - 1, y) + orgPel.at(x - 1, y + 1));
					  sobelBotL += ((x_sum) * (x_sum)) + ((y_sum) * (y_sum));
				  }
			  }
		  }
		  sobel += sobelBotL;
		  sobelBotL = sobelBotL / ((halfH - 2) * (halfW - 2));
		  gradHor += gradHorBotL;   gradVer += gradVerBotL;
		  gradHorBotL = gradHorBotL / ((halfH - 1) * (halfW - 1));
		  gradVerBotL = gradVerBotL / ((halfH - 1) * (halfW - 1));


		  double sobelBotR = 0.0;
		  double gradHorBotR = 0.0, gradVerBotR = 0.0;
		  for (int y = halfH; y < ht - 1; y++)
		  {
			  for (int x = halfW; x < wd - 1; x++)
			  {
				  gradHorBotR += abs(orgPel.at(x + 1, y) - orgPel.at(x, y));
				  gradVerBotR += abs(orgPel.at(x, y + 1) - orgPel.at(x, y));
				  if (x >= 1 && y >= 1)
				  {
					  double x_sum = (orgPel.at(x - 1, y + 1) + 2 * orgPel.at(x, y + 1) + orgPel.at(x + 1, y + 1))
						  - (orgPel.at(x - 1, y - 1) + 2 * orgPel.at(x, y - 1) + orgPel.at(x + 1, y - 1));
					  double y_sum = (orgPel.at(x + 1, y - 1) + 2 * orgPel.at(x + 1, y) + orgPel.at(x + 1, y + 1))
						  - (orgPel.at(x - 1, y - 1) + 2 * orgPel.at(x - 1, y) + orgPel.at(x - 1, y + 1));
					  sobelBotR += ((x_sum) * (x_sum)) + ((y_sum) * (y_sum));
				  }
			  }
		  }
		  sobel += sobelBotR;
		  sobelBotR = sobelBotR / ((halfH - 2) * (halfW - 2));
		  gradHor += gradHorBotR;   gradVer += gradVerBotR;
		  gradHorBotR = gradHorBotR / ((halfH - 1) * (halfW - 1));
		  gradVerBotR = gradVerBotR / ((halfH - 1) * (halfW - 1));


		  for (int y = 0; y < ht - 1; y++)
		  {
			  gradHor += abs(orgPel.at(halfW , y) - orgPel.at(halfW - 1, y));
			  if (y >= 1)
			  {
				  double x_sum = (orgPel.at(halfW - 1, y + 1) + 2 * orgPel.at(halfW, y + 1) + orgPel.at(halfW + 1, y + 1))
					  - (orgPel.at(halfW - 1, y - 1) + 2 * orgPel.at(halfW, y - 1) + orgPel.at(halfW + 1, y - 1));
				  double y_sum = (orgPel.at(halfW + 1, y - 1) + 2 * orgPel.at(halfW + 1, y) + orgPel.at(halfW + 1, y + 1))
					  - (orgPel.at(halfW - 1, y - 1) + 2 * orgPel.at(halfW - 1, y) + orgPel.at(halfW - 1, y + 1));
				  sobel += ((x_sum) * (x_sum)) + ((y_sum) * (y_sum));
			  }
		  }
		  for (int x = 0; x < wd - 1; x++)
		  {
			  gradVer += abs(orgPel.at(x, halfH) - orgPel.at(x, halfH - 1));
			  if (x >= 1)
			  {
				  double x_sum = (orgPel.at(x - 1, halfH + 1) + 2 * orgPel.at(x, halfH + 1) + orgPel.at(x + 1, halfH + 1))
					  - (orgPel.at(x - 1, halfH - 1) + 2 * orgPel.at(x, halfH - 1) + orgPel.at(x + 1, halfH - 1));
				  double y_sum = (orgPel.at(x + 1, halfH - 1) + 2 * orgPel.at(x + 1, halfH) + orgPel.at(x + 1, halfH + 1))
					  - (orgPel.at(x - 1, halfH - 1) + 2 * orgPel.at(x - 1, halfH) + orgPel.at(x - 1, halfH + 1));
				  sobel += ((x_sum) * (x_sum)) + ((y_sum) * (y_sum));
			  }
		  }
		  gradHor = gradHor / ((ht - 1) * (wd - 1));
		  gradVer = gradVer / ((ht - 1) * (wd - 1));
		  sobel = sobel / ((ht - 2) * (wd - 2));

		  double gradRatio = gradHor / gradVer;

		  //const CPelBuf refPel = cs.slice->getRefPic(REF_PIC_LIST_0, 0)->getRecoBuf(cs.area.blocks[COMPONENT_Y]);
		  //int distScale = (cs.slice->getPOC() - cs.slice->getRefPic(REF_PIC_LIST_0, 0)->getPOC());
		  Mv* mvArray = cs.slice->getPic()->getMvArray();  
		  Mv* mvPtr;
		  double sumX = 0, squaredSumX = 0, sumY = 0, squaredSumY = 0;

		  int mvX[32][32], mvY[32][32];
		  int* sadArray = cs.slice->getPic()->getSADErr();
		  int* sadPtr;
		  int sadError[32][32];
		  double sumSAD = 0, squaredSumSAD = 0;
		  double squaredMul = 0;// mvMul = 0;
		  int x_cor = currArea.lumaPos().x, y_cor = currArea.lumaPos().y, picWd = cs.slice->getPic()->lwidth(), picHt = cs.slice->getPic()->lheight();

		  for (int y = 0; y < ht; y = y + 4)
		  {
			  for (int x = 0; x < wd; x = x + 4)
			  {
				  mvPtr = mvArray + ((y_cor + y) / 4) * (picWd / 4) + ((x_cor + x) / 4);
				  int bestX = (*mvPtr).getHor(), bestY = (*mvPtr).getVer();

				  sumX += bestX;  squaredSumX += (bestX * bestX);  sumY += bestY;  squaredSumY += (bestY * bestY);
				  mvX[y / 4][x / 4] = bestX;  mvY[y / 4][x / 4] = bestY;

				  sadPtr = sadArray + ((y_cor + y) / 4) * (picWd / 4) + ((x_cor + x) / 4);
				  int bestSAD = (*sadPtr);

				  sumSAD += bestSAD;  squaredSumSAD += (bestSAD * bestSAD);
				  sadError[y / 4][x / 4] = bestSAD;
				  squaredMul += (bestX * bestY) * (bestX * bestY);
			  }
		  }


		  double varMVX = ((double)(squaredSumX * 16) / (double)(ht * wd)) - (((double)(sumX * 16) / (double)(ht * wd)) * ((double)(sumX * 16) / (double)(ht * wd)));
		  double varMVY = ((double)(squaredSumY * 16) / (double)(ht * wd)) - (((double)(sumY * 16) / (double)(ht * wd)) * ((double)(sumY * 16) / (double)(ht * wd)));
		  //double VarMv = varMVX + varMVY;

		  double varSAD = ((double)(squaredSumSAD * 16) / (double)(ht * wd)) - (((double)(sumSAD * 16) / (double)(ht * wd)) * ((double)(sumSAD * 16) / (double)(ht * wd)));
		  double aveSAD = (double)(sumSAD * 16) / (double)(halfH * halfW);

		  double widthFactor = (double)picWd / (double)416;
		  double heightFactor = (double)picHt / (double)240;

		  double varMVXScaled = varMVX / (widthFactor * widthFactor);
		  double varMVYScaled = varMVY / (heightFactor * heightFactor);
		  double VarMvScaled = varMVXScaled + varMVYScaled;

		  double a = squaredSumX, b = squaredMul, c = squaredMul, d = squaredSumY;
		  double eigenDifference = (((a + d) * (a + d)) - (4 * (a * d - b * c))) / ((a + d) * (a + d));

		  double aveX = (double)(sumX * 16) / (double)(ht * wd);
		  double aveY = (double)(sumY * 16) / (double)(ht * wd);

		  sumX = 0, squaredSumX = 0, sumY = 0, squaredSumY = 0;
		  sumSAD = 0, squaredSumSAD = 0, squaredMul = 0;
		  for (int ii = 0; ii < ((halfH / 4)); ii++)
		  {
			  for (int jj = 0; jj < ((halfW / 4)); jj++)
			  {
				  sumX += mvX[ii][jj]; sumY += mvY[ii][jj];
				  squaredSumX += (mvX[ii][jj] * mvX[ii][jj]); squaredSumY += (mvY[ii][jj] * mvY[ii][jj]);
				  sumSAD += sadError[ii][jj];
				  squaredSumSAD += (sadError[ii][jj] * sadError[ii][jj]);
				  squaredMul += (mvX[ii][jj] * mvY[ii][jj]) * (mvX[ii][jj] * mvY[ii][jj]);
			  }
		  }
		  varMVX = ((double)(squaredSumX * 16) / (double)(halfH * halfW)) - (((double)(sumX * 16) / (double)(halfH * halfW)) * ((double)(sumX * 16) / (double)(halfH * halfW)));
		  varMVY = ((double)(squaredSumY * 16) / (double)(halfH * halfW)) - (((double)(sumY * 16) / (double)(halfH * halfW)) * ((double)(sumY * 16) / (double)(halfH * halfW)));
		  //double varMVTopL = varMVX + varMVY;

		  double varSADTopL = ((double)(squaredSumSAD * 16) / (double)(halfH * halfW)) - (((double)(sumSAD * 16) / (double)(halfH * halfW)) * ((double)(sumSAD * 16) / (double)(halfH * halfW)));
		  double aveSADTopL = (double)(sumSAD * 16) / (double)(halfH * halfW);

		  varMVXScaled = varMVX / (widthFactor * widthFactor);
		  varMVYScaled = varMVY / (heightFactor * heightFactor);
		  double varMVTopLScaled = varMVXScaled + varMVYScaled;

		  //double _squaredSumSADTopL = squaredSumSAD, _sumSADTopL = sumSAD;
		  double _squaredSumXTopL = squaredSumX, _squaredSumYTopL = squaredSumY, _sumXTopL = sumX, _sumYTopL = sumY;


		  aveX = 16 * sumX / (halfH * halfW);   aveY = 16 * sumY / (halfH * halfW);
		  double aveMVTopL = abs(aveX) + abs(aveY);
		  double aveMVTopLScaled = aveMVTopL / widthFactor;

		  sumX = 0, squaredSumX = 0, sumY = 0, squaredSumY = 0;
		  sumSAD = 0, squaredSumSAD = 0, squaredMul = 0;
		  for (int ii = 0; ii < ((halfH / 4)); ii++)
		  {
			  for (int jj = (halfW / 4); jj < (wd / 4); jj++)
			  {
				  sumX += mvX[ii][jj]; sumY += mvY[ii][jj];
				  squaredSumX += (mvX[ii][jj] * mvX[ii][jj]); squaredSumY += (mvY[ii][jj] * mvY[ii][jj]);
				  sumSAD += sadError[ii][jj];
				  squaredSumSAD += (sadError[ii][jj] * sadError[ii][jj]);
				  squaredMul += (mvX[ii][jj] * mvY[ii][jj]) * (mvX[ii][jj] * mvY[ii][jj]);
			  }
		  }
		  varMVX = ((double)(squaredSumX * 16) / (double)(halfH * halfW)) - (((double)(sumX * 16) / (double)(halfH * halfW)) * ((double)(sumX * 16) / (double)(halfH * halfW)));
		  varMVY = ((double)(squaredSumY * 16) / (double)(halfH * halfW)) - (((double)(sumY * 16) / (double)(halfH * halfW)) * ((double)(sumY * 16) / (double)(halfH * halfW)));
		  //double varMVTopR = varMVX + varMVY;

		  double varSADTopR = ((double)(squaredSumSAD * 16) / (double)(halfH * halfW)) - (((double)(sumSAD * 16) / (double)(halfH * halfW)) * ((double)(sumSAD * 16) / (double)(halfH * halfW)));
		  double aveSADTopR = (double)(sumSAD * 16) / (double)(halfH * halfW);

		  varMVXScaled = varMVX / (widthFactor * widthFactor);
		  varMVYScaled = varMVY / (heightFactor * heightFactor);
		  double varMVTopRScaled = varMVXScaled + varMVYScaled;

		  //double _squaredSumSADTopR = squaredSumSAD, _sumSADTopR = sumSAD;
		  double _squaredSumXTopR = squaredSumX, _squaredSumYTopR = squaredSumY, _sumXTopR = sumX, _sumYTopR = sumY;

		  aveX = 16 * sumX / (halfH * halfW);   aveY = 16 * sumY / (halfH * halfW);

		  double aveMVTopR = abs(aveX) + abs(aveY);
		  double aveMVTopRScaled = aveMVTopR / widthFactor;

		  sumX = 0, squaredSumX = 0, sumY = 0, squaredSumY = 0;
		  sumSAD = 0, squaredSumSAD = 0, squaredMul = 0;
		  for (int ii = (halfH / 4); ii < (ht / 4); ii++)
		  {
			  for (int jj = 0; jj < ((halfW / 4)); jj++)
			  {
				  sumX += mvX[ii][jj]; sumY += mvY[ii][jj];
				  squaredSumX += (mvX[ii][jj] * mvX[ii][jj]); squaredSumY += (mvY[ii][jj] * mvY[ii][jj]);
				  sumSAD += sadError[ii][jj];
				  squaredSumSAD += (sadError[ii][jj] * sadError[ii][jj]);
				  squaredMul += (mvX[ii][jj] * mvY[ii][jj]) * (mvX[ii][jj] * mvY[ii][jj]);
			  }
		  }
		  varMVX = ((double)(squaredSumX * 16) / (double)(halfH * halfW)) - (((double)(sumX * 16) / (double)(halfH * halfW)) * ((double)(sumX * 16) / (double)(halfH * halfW)));
		  varMVY = ((double)(squaredSumY * 16) / (double)(halfH * halfW)) - (((double)(sumY * 16) / (double)(halfH * halfW)) * ((double)(sumY * 16) / (double)(halfH * halfW)));
		  //double varMVBotL = varMVX + varMVY;

		  double varSADBotL = ((double)(squaredSumSAD * 16) / (double)(halfH * halfW)) - (((double)(sumSAD * 16) / (double)(halfH * halfW)) * ((double)(sumSAD * 16) / (double)(halfH * halfW)));
		  double aveSADBotL = (double)(sumSAD * 16) / (double)(halfH * halfW);

		  varMVXScaled = varMVX / (widthFactor * widthFactor);
		  varMVYScaled = varMVY / (heightFactor * heightFactor);
		  double varMVBotLScaled = varMVXScaled + varMVYScaled;

		  //double _squaredSumSADBotL = squaredSumSAD, _sumSADBotL = sumSAD;
		  double _squaredSumXBotL = squaredSumX, _squaredSumYBotL = squaredSumY, _sumXBotL = sumX, _sumYBotL = sumY;

		  aveX = 16 * sumX / (halfH * halfW);   aveY = 16 * sumY / (halfH * halfW);
		  double aveMVBotL = abs(aveX) + abs(aveY);
		  double aveMVBotLScaled = aveMVBotL / widthFactor;
		  sumX = 0, squaredSumX = 0, sumY = 0, squaredSumY = 0;
		  sumSAD = 0, squaredSumSAD = 0, squaredMul = 0;
		  for (int ii = (halfH / 4); ii < (ht / 4); ii++)
		  {
			  for (int jj = (halfW / 4); jj < (wd / 4); jj++)
			  {
				  sumX += mvX[ii][jj]; sumY += mvY[ii][jj];
				  squaredSumX += (mvX[ii][jj] * mvX[ii][jj]); squaredSumY += (mvY[ii][jj] * mvY[ii][jj]);
				  sumSAD += sadError[ii][jj];
				  squaredSumSAD += (sadError[ii][jj] * sadError[ii][jj]);
				  squaredMul += (mvX[ii][jj] * mvY[ii][jj]) * (mvX[ii][jj] * mvY[ii][jj]);
			  }
		  }
		  varMVX = ((double)(squaredSumX * 16) / (double)(halfH * halfW)) - (((double)(sumX * 16) / (double)(halfH * halfW)) * ((double)(sumX * 16) / (double)(halfH * halfW)));
		  varMVY = ((double)(squaredSumY * 16) / (double)(halfH * halfW)) - (((double)(sumY * 16) / (double)(halfH * halfW)) * ((double)(sumY * 16) / (double)(halfH * halfW)));
		  //double varMVBotR = varMVX + varMVY;

		  double varSADBotR = ((double)(squaredSumSAD * 16) / (double)(halfH * halfW)) - (((double)(sumSAD * 16) / (double)(halfH * halfW)) * ((double)(sumSAD * 16) / (double)(halfH * halfW)));
		  double aveSADBotR = (double)(sumSAD * 16) / (double)(halfH * halfW);

		  varMVXScaled = varMVX / (widthFactor * widthFactor);
		  varMVYScaled = varMVY / (heightFactor * heightFactor);
		  double varMVBotRScaled = varMVXScaled + varMVYScaled;

		  //double _squaredSumSADBotR = squaredSumSAD, _sumSADBotR = sumSAD;
		  double _squaredSumXBotR = squaredSumX, _squaredSumYBotR = squaredSumY, _sumXBotR = sumX, _sumYBotR = sumY;

		  aveX = 16 * sumX / (halfH * halfW);   aveY = 16 * sumY / (halfH * halfW);
		  double aveMVBotR = abs(aveX) + abs(aveY);
		  double aveMVBotRScaled = aveMVBotR / widthFactor;
		  double aveMVScaled = (aveMVTopRScaled + aveMVTopLScaled + aveMVBotLScaled + aveMVBotRScaled) / 4;

		  double ratio2HvarPix1 = (double)8 * (_squaredSumPixTopL + _squaredSumPixBotL) / (double)(halfH * halfW) - (double)(8 * (_sumPixTopL + _sumPixBotL) / (double)(halfH * halfW)) * (8 * (_sumPixTopL + _sumPixBotL) / (double)(halfH * halfW));
		  double ratio2HvarPix2 = (double)8 * (_squaredSumPixTopR + _squaredSumPixBotR) / (double)(halfH * halfW) - (double)(8 * (_sumPixTopR + _sumPixBotR) / (double)(halfH * halfW)) * (8 * (_sumPixTopR + _sumPixBotR) / (double)(halfH * halfW));
		  double ratio2HVarPix = abs(ratio2HvarPix1 / ratio2HvarPix2);

		  ratio2HvarPix1 = (double)8 * (_squaredSumPixTopL + _squaredSumPixTopR) / (double)(halfH * halfW) - (double)(8 * (_sumPixTopL + _sumPixTopR) / (double)(halfH * halfW)) * (8 * (_sumPixTopL + _sumPixTopR) / (double)(halfH * halfW));
		  ratio2HvarPix2 = (double)8 * (_squaredSumPixBotL + _squaredSumPixBotR) / (double)(halfH * halfW) - (double)(8 * (_sumPixBotL + _sumPixBotR) / (double)(halfH * halfW)) * (8 * (_sumPixBotL + _sumPixBotR) / (double)(halfH * halfW));
		  double ratio2VVarPix = abs(ratio2HvarPix1 / ratio2HvarPix2);

		  double ratio2HGrad = abs((gradHorTopL / gradVerTopL) + (gradHorBotL / gradVerBotL)) / abs((gradHorTopR / gradVerTopR) + (gradHorBotR / gradVerBotR));
		  double ratio2VGrad = abs((gradHorTopL / gradVerTopL) + (gradHorTopR / gradVerTopR)) / abs((gradHorBotL / gradVerBotL) + (gradHorBotR / gradVerBotR));

		  double ratio2HVarMVScaledX1 = (double)8 * (_squaredSumXTopL + _squaredSumXBotL) / (double)(halfH * halfW) - (double)(8 * (_sumXTopL + _sumXBotL) / (double)(halfH * halfW)) * (8 * (_sumXTopL + _sumXBotL) / (double)(halfH * halfW));
		  double ratio2HVarMVScaledY1 = (double)8 * (_squaredSumYTopL + _squaredSumYBotL) / (double)(halfH * halfW) - (double)(8 * (_sumYTopL + _sumYBotL) / (double)(halfH * halfW)) * (8 * (_sumYTopL + _sumYBotL) / (double)(halfH * halfW));
		  double ratio2HVarMVScaledX2 = (double)8 * (_squaredSumXTopR + _squaredSumXBotR) / (double)(halfH * halfW) - (double)(8 * (_sumXTopR + _sumXBotR) / (double)(halfH * halfW)) * (8 * (_sumXTopR + _sumXBotR) / (double)(halfH * halfW));
		  double ratio2HVarMVScaledY2 = (double)8 * (_squaredSumYTopR + _squaredSumYBotR) / (double)(halfH * halfW) - (double)(8 * (_sumYTopR + _sumYBotR) / (double)(halfH * halfW)) * (8 * (_sumYTopR + _sumYBotR) / (double)(halfH * halfW));		  
		  double ratio2HVarMVScaled = abs(ratio2HVarMVScaledX1 / ratio2HVarMVScaledX2) + abs(ratio2HVarMVScaledY1 / ratio2HVarMVScaledY2);

		  ratio2HVarMVScaledX1 = (double)8 * (_squaredSumXTopL + _squaredSumXTopR) / (double)(halfH * halfW) - (double)(8 * (_sumXTopL + _sumXTopR) / (double)(halfH * halfW)) * (8 * (_sumXTopL + _sumXTopR) / (double)(halfH * halfW));
		  ratio2HVarMVScaledY1 = (double)8 * (_squaredSumYTopL + _squaredSumYTopR) / (double)(halfH * halfW) - (double)(8 * (_sumYTopL + _sumYTopR) / (double)(halfH * halfW)) * (8 * (_sumYTopL + _sumYTopR) / (double)(halfH * halfW));
		  ratio2HVarMVScaledX2 = (double)8 * (_squaredSumXBotL + _squaredSumXBotR) / (double)(halfH * halfW) - (double)(8 * (_sumXBotL + _sumXBotR) / (double)(halfH * halfW)) * (8 * (_sumXBotL + _sumXBotR) / (double)(halfH * halfW));
		  ratio2HVarMVScaledY2 = (double)8 * (_squaredSumYBotL + _squaredSumYBotR) / (double)(halfH * halfW) - (double)(8 * (_sumYBotL + _sumYBotR) / (double)(halfH * halfW)) * (8 * (_sumYBotL + _sumYBotR) / (double)(halfH * halfW));
		  double ratio2VVarMVScaled = abs(ratio2HVarMVScaledX1 / ratio2HVarMVScaledX2) + abs(ratio2HVarMVScaledY1 / ratio2HVarMVScaledY2);
		  //double ratio2HAveMVScaled = abs((aveMVTopLScaled + aveMVBotLScaled) / (aveMVTopRScaled + aveMVBotRScaled));
		  //double ratio2VAveMVScaled = abs((aveMVTopLScaled + aveMVTopRScaled) / (aveMVBotLScaled + aveMVBotRScaled));



		  double ratio2HaveSAD = abs((aveSADTopL + aveSADBotL) / (aveSADTopR + aveSADBotR));
		  double ratio2VaveSAD = abs((aveSADTopL + aveSADTopR) / (aveSADBotL + aveSADBotR));


		  double ratio2HSobel = abs((sobelTopL + sobelTopR) / (sobelBotL + sobelBotR));
		  double ratio2VSobel = abs((sobelTopL + sobelBotL) / (sobelTopR + sobelBotR));

		  double ratio2HVVarMVScaled = ratio2HVarMVScaled / ratio2VVarMVScaled;
		  //double ratio2HVAveMVScaled = ratio2HAveMVScaled / ratio2VAveMVScaled;

		  double ratio2HVSobel = ratio2HSobel / ratio2VSobel;

			const CodingStructure* bestCS = cuECtx.bestCS;
			bool isInter = false, isMerge = false, isGeo = false, isIntra = false;
			if (bestCS != NULL && !bestCS->cus.empty())
			{
				CodingUnit* bestnonSplitcu = bestCS->cus[0];
				bool isIntra = CU::isIntra(*bestnonSplitcu);
				cuECtx.set(IS_NON_SPLIT_INTRA, isIntra);
				if (!isIntra)
				{
					isInter = CU::isInter(*bestnonSplitcu) && !bestnonSplitcu->firstPU->mergeFlag;
					isMerge = CU::isInter(*bestnonSplitcu) && bestnonSplitcu->firstPU->mergeFlag && !bestnonSplitcu->geoFlag;
					isGeo = CU::isInter(*bestnonSplitcu) && bestnonSplitcu->geoFlag;
				}
				else
				{
					isInter = false;
					isMerge = false;
					isGeo = false;
				}
			}


			int tLayer = cs.slice->getPic()->layer;
			float qTMTTFeatures[34] = { (float)tLayer, (float)qp, (float)var, (float)gradHor, (float)gradVer, (float)gradRatio, (float)varTopL, (float)varTopR, (float)varBotL, (float)varBotR,
											(float)VarMvScaled, (float)varMVTopLScaled, (float)varMVTopRScaled, (float)varMVBotLScaled, (float)varMVBotRScaled, (float)eigenDifference, (float)aveSAD, (float)varSAD, (float)varSADTopL, (float)varSADTopR, (float)varSADBotL, (float)varSADBotR, (float)sobelTopL, (float)sobelTopR, (float)sobelBotL, (float)sobelBotR, (float)ratio2HGrad, (float)ratio2VGrad, (float)ratio2HVarMVScaled, (float)ratio2VVarMVScaled, (float)ratio2HVVarMVScaled, (float)isIntra, (float)isInter, (float)isMerge };
		
			float horVerFeatures[45] = { (float)tLayer, (float)qp, (float)var, (float)gradHor, (float)gradVer, (float)gradRatio, (float)varTopL, (float)varTopR, (float)varBotL, (float)varBotR,
											(float)VarMvScaled, (float)varMVTopLScaled, (float)varMVTopRScaled, (float)varMVBotLScaled, (float)varMVBotRScaled, (float)aveMVScaled, (float)aveMVTopLScaled, (float)aveMVTopRScaled, (float)aveMVBotLScaled, (float)aveMVBotRScaled, (float)aveSAD, (float)varSAD, (float)varSADTopL, (float)varSADTopR, (float)varSADBotL, (float)varSADBotR, (float)sobelTopL, (float)sobelTopR, (float)sobelBotL, (float)sobelBotR, (float)ratio2HVarPix, (float)ratio2VVarPix, (float)ratio2HGrad, (float)ratio2VGrad, (float)ratio2HVarMVScaled, (float)ratio2VVarMVScaled, (float)ratio2HaveSAD, (float)ratio2VaveSAD, (float)ratio2HSobel, (float)ratio2VSobel, (float)ratio2HVSobel, (float)isIntra, (float)isInter, (float)isMerge, (float)isGeo };



		bool noZeroDenom =  gradVerBotR != 0.0 && gradVerBotL != 0.0 && gradVerTopR != 0.0 && gradVerTopL != 0.0 && gradVer != 0.0;
		if (!noZeroDenom)
		{
			cuECtx.set(NO_SPLIT_FLAG, 2);
			cuECtx.set(QT_FLAG, 2);
			cuECtx.set(HOR_FLAG, 2);
		}
		else
		{

#if COLLECT_DATASET
      std::string nameFile = filename_arg.substr(filename_arg.find_last_of("/\\") + 1);
      std::string sTracingFile_feature = "split_features_" + nameFile.substr(0, nameFile.find_last_of(".")) + "_QP_" + to_string(qp_arg) + ".csv";

      FILE* m_trace_file_f = fopen( sTracingFile_feature.c_str(), "a+" );

      std::string format_head = "%d;%d;%d;%d;%d;%ld;%d;";
#else
			RandomForestClassfier rf;
#endif
			double noSplitFrac = 0.5;

#if INTIALIZE_TO_0_5
			int noSplitFlag = noSplitFrac > 0.50 ? 1 : (noSplitFrac < 0.50 ? 0 : 2);
#endif
#if INTIALIZE_TO_0_75
			int noSplitFlag = noSplitFrac > 0.75 ? 1 : (noSplitFrac < 0.25 ? 0 : 2);
#endif
#if INTIALIZE_TO_0_85
			int noSplitFlag = noSplitFrac > 0.85 ? 1 : (noSplitFrac < 0.15 ? 0 : 2);
#endif
#if INTIALIZE_TO_1_00
			int noSplitFlag = noSplitFrac > 0.975 ? 1 : (noSplitFrac < 0.025 ? 0 : 2);
#endif
#if INTIALIZE_TO_0_90
			int noSplitFlag = noSplitFrac > 0.90 ? 1 : (noSplitFrac < 0.10 ? 0 : 2);
#endif
#if INTIALIZE_TO_0_95
			int noSplitFlag = noSplitFrac > 0.95 ? 1 : (noSplitFrac < 0.05 ? 0 : 2);
#endif

			//int qTFlag = 2, horFlag = 2, btFlag = 2;
			int qTFlag = 2, horFlag = 2;
			cuECtx.set(NO_SPLIT_FLAG, noSplitFlag);
			if (noSplitFlag != 1)
			{
#if COLLECT_DATASET
        double qTFrac = 0.5;
        if (wd == ht && wd != 8){
        WriteFormatted_features(m_trace_file_f, format_head.c_str(), cs.slice->getPOC(), ht, wd, partitioner.currArea().lx(), partitioner.currArea().ly(), partitioner.getSplitSeries(), 0);  
        for(int i = 0; i < 34; i++){
         WriteFormatted_features(m_trace_file_f, "%f;", qTMTTFeatures[i]);
        }
        WriteFormatted_features(m_trace_file_f, "\n");
      }
#else
				double qTFrac = (wd == ht) ? rf.predictQTMTT(qTMTTFeatures, wd, ht) : 0.5;
#endif

#if PCA_BASED_THRESHOLDS
				qTFlag = qTFrac > pcaThresholdQT ? 1 : (qTFrac < (1 - pcaThresholdQT) ? 0 : 2);
#else
#if INTIALIZE_TO_0_5
				qTFlag = qTFrac > 0.50 ? 1 : (qTFrac < 0.50 ? 0 : 2);
#endif
#if INTIALIZE_TO_0_75
				qTFlag = qTFrac > 0.75 ? 1 : (qTFrac < 0.25 ? 0 : 2);
#endif
#if INTIALIZE_TO_0_85
				qTFlag = qTFrac > 0.85 ? 1 : (qTFrac < 0.15 ? 0 : 2);
#endif
#if INTIALIZE_TO_1_00
				qTFlag = qTFrac > 0.975 ? 1 : (qTFrac < 0.025 ? 0 : 2);
#endif
#if INTIALIZE_TO_0_90
				qTFlag = qTFrac > 0.90 ? 1 : (qTFrac < 0.10 ? 0 : 2);
#endif
#if INTIALIZE_TO_0_95
				qTFlag = qTFrac > 0.95 ? 1 : (qTFrac < 0.05 ? 0 : 2);
#endif
#endif
				cuECtx.set(QT_FLAG, qTFlag);
				bool zeroDenom2 = (aveMVTopRScaled + aveMVBotRScaled) == 0.0 || (aveMVBotLScaled + aveMVBotRScaled) == 0.0 || (aveSADTopR + aveSADBotR) == 0.0 || (aveSADBotL + aveSADBotR) == 0.0 || (sobelBotL + sobelBotR) == 0.0 || (sobelTopR + sobelBotR) == 0.0 || ratio2VSobel == 0.0;
				if (qTFlag != 1 && !zeroDenom2)
				{

#if COLLECT_DATASET
          double horFrac = 0.5;
          WriteFormatted_features(m_trace_file_f, format_head.c_str(), cs.slice->getPOC(), ht, wd, partitioner.currArea().lx(), partitioner.currArea().ly(), partitioner.getSplitSeries(), 1);  
          for(int i = 0; i < 45; i++){
           WriteFormatted_features(m_trace_file_f, "%f;", horVerFeatures[i]);
          }
          WriteFormatted_features(m_trace_file_f, "\n");
#else
					double horFrac = rf.predictHorVer(horVerFeatures, wd, ht);
#endif

#if PCA_BASED_THRESHOLDS
					horFlag = horFrac > pcaThresholdHor ? 1 : (horFrac < (1 - pcaThresholdHor) ? 0 : 2);;
#else
#if INTIALIZE_TO_0_5
					horFlag = horFrac > 0.50 ? 1 : (horFrac < 0.50 ? 0 : 2);
#endif
#if INTIALIZE_TO_0_75
					horFlag = horFrac > 0.75 ? 1 : (horFrac < 0.25 ? 0 : 2);
#endif
#if INTIALIZE_TO_0_85
					horFlag = horFrac > 0.85 ? 1 : (horFrac < 0.15 ? 0 : 2);
#endif
#if INTIALIZE_TO_1_00
					horFlag = horFrac > 0.975 ? 1 : (horFrac < 0.025 ? 0 : 2);
#endif
#if INTIALIZE_TO_0_90
					horFlag = horFrac > 0.90 ? 1 : (horFrac < 0.10 ? 0 : 2);
#endif
#if INTIALIZE_TO_0_95
					horFlag = horFrac > 0.95 ? 1 : (horFrac < 0.05 ? 0 : 2);
#endif
#endif
					cuECtx.set(HOR_FLAG, horFlag);
				}
#if COLLECT_DATASET
        fclose(m_trace_file_f);
#endif
			}
		}


	  }

	  if (encTestmode.type != ETM_POST_DONT_SPLIT)
	  return false;
  }
  else if (encTestmode.type == IS_FIRST_MODE)
  {
	return false;
  }
#endif

  // Fast checks, partitioning depended
#if !JVET_Q0806
  if (cuECtx.isHashPerfectMatch && encTestmode.type != ETM_MERGE_SKIP && encTestmode.type != ETM_INTER_ME && encTestmode.type != ETM_AFFINE && encTestmode.type != ETM_MERGE_TRIANGLE)
#else
  if (cuECtx.isHashPerfectMatch && encTestmode.type != ETM_MERGE_SKIP && encTestmode.type != ETM_INTER_ME && encTestmode.type != ETM_AFFINE && encTestmode.type != ETM_MERGE_GEO)
#endif
  {
    return false;
  }

  // if early skip detected, skip all modes checking but the splits
  if( cuECtx.earlySkip && m_pcEncCfg->getUseEarlySkipDetection() && !isModeSplit( encTestmode ) && !( isModeInter( encTestmode ) ) )
  {
    return false;
  }

  const PartSplit implicitSplit = partitioner.getImplicitSplit( cs );
  const bool isBoundary         = implicitSplit != CU_DONT_SPLIT;

  /*const CompArea& currArea1 = partitioner.currArea().Y();
  int x_cor = currArea.lumaPos().x, y_cor = currArea.lumaPos().y;
  if (x_cor == 384 && y_cor == 384 && ht == 64 && wd == 64 && cs.slice->getPOC() == 14)
  {
	  std::cout << "Testing Mode: " << encTestmode.type << std::endl;
  }*/

#if DISABLE_RF_IF_EMPTY_CU_WHEN_FULL
  if (encTestmode.type == ETM_POST_DONT_SPLIT)
  {
	  const CodingStructure* bestCS = cuECtx.bestCS;
	  cuECtx.set(EMPTY_CU_WHEN_FULL, (bestCS == NULL || (bestCS != NULL && bestCS->cus.empty())) ? true : false);
  }
#endif

#if FEATURE_TEST 

#if DISABLE_RF_IF_EMPTY_CU_WHEN_FULL
	  if(!cuECtx.get<bool>(EMPTY_CU_WHEN_FULL))
	  {
#endif
		  if (cs.slice->getSliceType() != I_SLICE && sizeIndex != 28)
		  {
			  //int noSplitFlag = cuECtx.get<int>(NO_SPLIT_FLAG);
			  int qTFlag = cuECtx.get<int>(QT_FLAG);
			  int horFlag = cuECtx.get<int>(HOR_FLAG);
			  //noSplitFlag = 2;

#if MORE_RESTRICTIVE_SKIP
			  if (qTFlag == 1 && encTestmode.type != ETM_SPLIT_QT && isModeSplit(encTestmode) && cuECtx.get<bool>(DID_QUAD_SPLIT))
#else
			  if (qTFlag == 1 && encTestmode.type != ETM_SPLIT_QT && isModeSplit(encTestmode))
#endif
			  {
				  {
					  cuECtx.set(DID_HORZ_SPLIT, false);
					  cuECtx.set(DID_VERT_SPLIT, false);
					  cuECtx.set(DO_TRIH_SPLIT, false);
					  cuECtx.set(DO_TRIV_SPLIT, false);
					  return false;
				  }
			  }
#if MORE_RESTRICTIVE_SKIP
			  if (qTFlag == 0 && isModeSplit(encTestmode) && (cuECtx.get<bool>(DID_HORZ_SPLIT) || cuECtx.get<bool>(DID_VERT_SPLIT) || cuECtx.get<bool>(DO_TRIH_SPLIT) || cuECtx.get<bool>(DO_TRIV_SPLIT)))
#else
			  if (!qTFlag && isModeSplit(encTestmode))
#endif
			  {
				  if (encTestmode.type == ETM_SPLIT_QT)
				  {
					  cuECtx.set(DID_QUAD_SPLIT, false);
					  return false;
				  }
			  }
#if MORE_RESTRICTIVE_SKIP
			  if (horFlag == 1 && (encTestmode.type == ETM_SPLIT_BT_V || encTestmode.type == ETM_SPLIT_TT_V) && (cuECtx.get<bool>(DID_HORZ_SPLIT) || cuECtx.get<bool>(DO_TRIH_SPLIT)))
#else
			  if (horFlag == 1 && (encTestmode.type == ETM_SPLIT_BT_V || encTestmode.type == ETM_SPLIT_TT_V))
#endif
			  {
				  {
					  cuECtx.set(DID_VERT_SPLIT, false);
					  cuECtx.set(DO_TRIV_SPLIT, false);
					  return false;
				  }
			  }
#if MORE_RESTRICTIVE_SKIP
			  if (horFlag == 0 && (encTestmode.type == ETM_SPLIT_BT_H || encTestmode.type == ETM_SPLIT_TT_H) && (cuECtx.get<bool>(DID_VERT_SPLIT) || cuECtx.get<bool>(DO_TRIV_SPLIT)))
#else
			  if (!horFlag && (encTestmode.type == ETM_SPLIT_BT_H || encTestmode.type == ETM_SPLIT_TT_H))
#endif
			  {
				  //if (!(isBoundary && getPartSplit(encTestmode) == implicitSplit))
				  {
					  cuECtx.set(DID_HORZ_SPLIT, false);
					  cuECtx.set(DO_TRIH_SPLIT, false);
					  return false;
				  }
			  }
		  }
#if DISABLE_RF_IF_EMPTY_CU_WHEN_FULL
	  }
#endif
#endif

  if( isBoundary && encTestmode.type != ETM_SPLIT_QT )
  {
    return getPartSplit( encTestmode ) == implicitSplit;
  }
  else if( isBoundary && encTestmode.type == ETM_SPLIT_QT )
  {
    return partitioner.canSplit( CU_QUAD_SPLIT, cs );
  }


#if REUSE_CU_RESULTS
  if( cuECtx.get<bool>( IS_REUSING_CU ) )
  {
    if( encTestmode.type == ETM_RECO_CACHED )
    {
      return true;
    }

    if( isModeNoSplit( encTestmode ) )
    {
      return false;
    }
  }

#endif
  const Slice&           slice       = *m_slice;
  const SPS&             sps         = *slice.getSPS();
  const uint32_t             numComp     = getNumberValidComponents( slice.getSPS()->getChromaFormatIdc() );
  const uint32_t             width       = partitioner.currArea().lumaSize().width;
  const CodingStructure *bestCS      = cuECtx.bestCS;
  const CodingUnit      *bestCU      = cuECtx.bestCU;
  const EncTestMode      bestMode    = bestCS ? getCSEncMode( *bestCS ) : EncTestMode();

  CodedCUInfo    &relatedCU          = getBlkInfo( partitioner.currArea() );

  if( cuECtx.minDepth > partitioner.currQtDepth && partitioner.canSplit( CU_QUAD_SPLIT, cs ) )
  {
    // enforce QT
    return encTestmode.type == ETM_SPLIT_QT;
  }
  else if( encTestmode.type == ETM_SPLIT_QT && cuECtx.maxDepth <= partitioner.currQtDepth )
  {
    // don't check this QT depth
    return false;
  }

  if( bestCS && bestCS->cus.size() == 1 )
  {
    // update the best non-split cost
    cuECtx.set( BEST_NON_SPLIT_COST, bestCS->cost );
  }

  if( encTestmode.type == ETM_INTRA )
  {
    if( getFastDeltaQp() )
    {
      if( cs.area.lumaSize().width > cs.pcv->fastDeltaQPCuMaxSize )
      {
        return false; // only check necessary 2Nx2N Intra in fast delta-QP mode
      }
    }

    if( m_pcEncCfg->getUseFastLCTU() && partitioner.currArea().lumaSize().area() > 4096 )
    {
      return false;
    }

    if (CS::isDualITree(cs) && (partitioner.currArea().lumaSize().width > 64 || partitioner.currArea().lumaSize().height > 64))
    {
      return false;
    }

    if (m_pcEncCfg->getUsePbIntraFast() && (!cs.slice->isIntra() || cs.slice->getSPS()->getIBCFlag()) && !interHadActive(cuECtx) && cuECtx.bestCU && !CU::isIntra(*cuECtx.bestCU))
    {
      return false;
    }

    // INTRA MODES
    if (cs.sps->getIBCFlag() && !cuECtx.bestTU)
      return true;
    if( partitioner.isConsIntra() && !cuECtx.bestTU )
    {
      return true;
    }
    if ( partitioner.currArea().lumaSize().width == 4 && partitioner.currArea().lumaSize().height == 4 && !slice.isIntra() && !cuECtx.bestTU )
    {
      return true;
    }
    if( !( slice.isIRAP() || bestMode.type == ETM_INTRA || !cuECtx.bestTU ||
      ((!m_pcEncCfg->getDisableIntraPUsInInterSlices()) && (!relatedCU.isInter || !relatedCU.isIBC) && (
                                         ( cuECtx.bestTU->cbf[0] != 0 ) ||
           ( ( numComp > COMPONENT_Cb ) && cuECtx.bestTU->cbf[1] != 0 ) ||
           ( ( numComp > COMPONENT_Cr ) && cuECtx.bestTU->cbf[2] != 0 )  // avoid very complex intra if it is unlikely
         ) ) ) )
    {
      return false;
    }
    if ((m_pcEncCfg->getIBCFastMethod() & IBC_FAST_METHOD_NOINTRA_IBCCBF0)
      && (bestMode.type == ETM_IBC || bestMode.type == ETM_IBC_MERGE)
      && (!cuECtx.bestCU->Y().valid() || cuECtx.bestTU->cbf[0] == 0)
      && (!cuECtx.bestCU->Cb().valid() || cuECtx.bestTU->cbf[1] == 0)
      && (!cuECtx.bestCU->Cr().valid() || cuECtx.bestTU->cbf[2] == 0))
    {
      return false;
    }
    if( lastTestMode().type != ETM_INTRA && cuECtx.bestCS && cuECtx.bestCU && interHadActive( cuECtx ) )
    {
      // Get SATD threshold from best Inter-CU
      if (!cs.slice->isIRAP() && m_pcEncCfg->getUsePbIntraFast() && !cs.slice->getDisableSATDForRD())
      {
        CodingUnit* bestCU = cuECtx.bestCU;
        if (bestCU && !CU::isIntra(*bestCU))
        {
          DistParam distParam;
          const bool useHad = true;
          m_pcRdCost->setDistParam( distParam, cs.getOrgBuf( COMPONENT_Y ), cuECtx.bestCS->getPredBuf( COMPONENT_Y ), cs.sps->getBitDepth( CHANNEL_TYPE_LUMA ), COMPONENT_Y, useHad );
          cuECtx.interHad = distParam.distFunc( distParam );
        }
      }
    }
#if JVET_Q0504_PLT_NON444
    if (bestMode.type == ETM_PALETTE && !slice.isIRAP() && partitioner.treeType == TREE_D && !(partitioner.currArea().lumaSize().width == 4 && partitioner.currArea().lumaSize().height == 4)) // inter slice
#else
    if (bestMode.type == ETM_PALETTE && !slice.isIRAP() && !( partitioner.currArea().lumaSize().width == 4 && partitioner.currArea().lumaSize().height == 4) ) // inter slice
#endif
    {
      return false;
    }
    if ( m_pcEncCfg->getUseFastISP() && relatedCU.relatedCuIsValid )
    {
      cuECtx.ispPredModeVal     = relatedCU.ispPredModeVal;
      cuECtx.bestDCT2NonISPCost = relatedCU.bestDCT2NonISPCost;
      cuECtx.relatedCuIsValid   = relatedCU.relatedCuIsValid;
      cuECtx.bestNonDCT2Cost    = relatedCU.bestNonDCT2Cost;
      cuECtx.bestISPIntraMode   = relatedCU.bestISPIntraMode;
    }
    return true;
  }
  else if (encTestmode.type == ETM_PALETTE)
  {
#if JVET_Q0629_REMOVAL_PLT_4X4
    if (partitioner.currArea().lumaSize().width > 64 || partitioner.currArea().lumaSize().height > 64
        || ((partitioner.currArea().lumaSize().width * partitioner.currArea().lumaSize().height <= 16) && (isLuma(partitioner.chType)) )
        || ((partitioner.currArea().chromaSize().width * partitioner.currArea().chromaSize().height <= 16) && (!isLuma(partitioner.chType)) && partitioner.isSepTree(cs) ) )
#else
    if (partitioner.currArea().lumaSize().width > 64 || partitioner.currArea().lumaSize().height > 64)
#endif            
    {
      return false;
    }
    const Area curr_cu = CS::getArea(cs, cs.area, partitioner.chType).blocks[getFirstComponentOfChannel(partitioner.chType)];
    try
    {
#if JVET_Q0504_PLT_NON444
      double stored_cost = slice.m_mapPltCost[isChroma(partitioner.chType)].at(curr_cu.pos()).at(curr_cu.size());
#else
      double stored_cost = slice.m_mapPltCost.at(curr_cu.pos()).at(curr_cu.size());
#endif
      if (bestMode.type != ETM_INVALID && stored_cost > cuECtx.bestCS->cost)
      {
        return false;
      }
    }
    catch (const std::out_of_range &)
    {
      // do nothing if no stored cost value was found.
    }
    return true;
  }
  else if (encTestmode.type == ETM_IBC || encTestmode.type == ETM_IBC_MERGE)
  {
    // IBC MODES
    return sps.getIBCFlag() && (partitioner.currArea().lumaSize().width < 128 && partitioner.currArea().lumaSize().height < 128);
  }
  else if( isModeInter( encTestmode ) )
  {
    // INTER MODES (ME + MERGE/SKIP)
    CHECK( slice.isIntra(), "Inter-mode should not be in the I-Slice mode list!" );

    if( getFastDeltaQp() )
    {
      if( encTestmode.type == ETM_MERGE_SKIP )
      {
        return false;
      }
      if( cs.area.lumaSize().width > cs.pcv->fastDeltaQPCuMaxSize )
      {
        return false; // only check necessary 2Nx2N Inter in fast deltaqp mode
      }
    }

    // --- Check if we can quit current mode using SAVE/LOAD coding history

    if( encTestmode.type == ETM_INTER_ME )
    {
      if( encTestmode.opts == ETO_STANDARD )
      {
        // NOTE: ETO_STANDARD is always done when early SKIP mode detection is enabled
#if !ESD_ADJUST
        if( !m_pcEncCfg->getUseEarlySkipDetection() )
#endif
        {
          if( relatedCU.isSkip || relatedCU.isIntra )
          {
            return false;
          }
        }
      }
      else if ((encTestmode.opts & ETO_IMV) != 0)
      {
        int imvOpt = (encTestmode.opts & ETO_IMV) >> ETO_IMV_SHIFT;

        if (imvOpt == 3 && cuECtx.get<double>(BEST_NO_IMV_COST) * 1.06 < cuECtx.get<double>(BEST_IMV_COST))
        {
          if ( !m_pcEncCfg->getUseAffineAmvr() )
          return false;
        }
      }
    }


    if ( encTestmode.type == ETM_AFFINE && relatedCU.isIntra )
    {
      return false;
    }
#if !JVET_Q0806
    if( encTestmode.type == ETM_MERGE_TRIANGLE && ( partitioner.currArea().lumaSize().area() < TRIANGLE_MIN_SIZE || relatedCU.isIntra ) )
    {
      return false;
    }
#else
    if( encTestmode.type == ETM_MERGE_GEO && ( partitioner.currArea().lwidth() < GEO_MIN_CU_SIZE || partitioner.currArea().lheight() < GEO_MIN_CU_SIZE  
                                            || partitioner.currArea().lwidth() > GEO_MAX_CU_SIZE || partitioner.currArea().lheight() > GEO_MAX_CU_SIZE
                                            || partitioner.currArea().lwidth() >= 8 * partitioner.currArea().lheight()
                                            || partitioner.currArea().lheight() >= 8 * partitioner.currArea().lwidth() ) )
    {
      return false;
    }
#endif
    return true;
  }
  else if( isModeSplit( encTestmode ) )
  {
    //////////////////////////////////////////////////////////////////////////
    // skip-history rule - don't split further if at least for three past levels
    //                     in the split tree it was found that skip is the best mode
    //////////////////////////////////////////////////////////////////////////
    int skipScore = 0;

    if ((!slice.isIntra() || slice.getSPS()->getIBCFlag()) && cuECtx.get<bool>(IS_BEST_NOSPLIT_SKIP))
    {
      for( int i = 2; i < m_ComprCUCtxList.size(); i++ )
      {
        if( ( m_ComprCUCtxList.end() - i )->get<bool>( IS_BEST_NOSPLIT_SKIP ) )
        {
          skipScore += 1;
        }
        else
        {
          break;
        }
      }
    }


    const PartSplit split = getPartSplit( encTestmode );
const CompArea& currArea = partitioner.currArea().Y();
	int cuHeight = currArea.height;
	int cuWidth = currArea.width;

#if TT_SPEEDUPS 
	if (bestCS && bestCS->cus.size() == 1)
	{
		CodingUnit *bestnonSplitcu = bestCS->cus[0];
		bool mode = CU::isInter(*bestnonSplitcu) && (!bestnonSplitcu->firstPU->mergeFlag || bestnonSplitcu->geoFlag);
		if (split == CU_TRIH_SPLIT)
		{
			bool size = (cuWidth == 8 && cuHeight == 64) || (cuWidth == 4 && cuHeight == 64) || (cuWidth == 16 && cuHeight == 64);
			if (!(mode && size))
				return false;
		}
		else if (split == CU_TRIV_SPLIT)
		{
			bool size = (cuWidth == 64 && cuHeight == 8) || (cuWidth == 64 && cuHeight == 4)  || (cuWidth == 64 && cuHeight == 16);
			if (!(mode && size))
				return false;
		}
	}
#endif

	if (!partitioner.canSplit(split, cs) || skipScore >= 2)
    {
      if( split == CU_HORZ_SPLIT ) cuECtx.set( DID_HORZ_SPLIT, false );
      if( split == CU_VERT_SPLIT ) cuECtx.set( DID_VERT_SPLIT, false );
      if( split == CU_QUAD_SPLIT ) cuECtx.set( DID_QUAD_SPLIT, false );

      return false;
    }


    if( m_pcEncCfg->getUseContentBasedFastQtbt() )
    {
      const CompArea& currArea = partitioner.currArea().Y();
      int cuHeight  = currArea.height;
      int cuWidth   = currArea.width;

      const bool condIntraInter = m_pcEncCfg->getIntraPeriod() == 1 ? ( partitioner.currBtDepth == 0 ) : ( cuHeight > 32 && cuWidth > 32 );

      if( cuWidth == cuHeight && condIntraInter && getPartSplit( encTestmode ) != CU_QUAD_SPLIT )
      {
        const CPelBuf bufCurrArea = cs.getOrgBuf( partitioner.currArea().block( COMPONENT_Y ) );

        double horVal = 0;
        double verVal = 0;
        double dupVal = 0;
        double dowVal = 0;

        const double th = m_pcEncCfg->getIntraPeriod() == 1 ? 1.2 : 1.0;

        unsigned j, k;

        for( j = 0; j < cuWidth - 1; j++ )
        {
          for( k = 0; k < cuHeight - 1; k++ )
          {
            horVal += abs( bufCurrArea.at( j + 1, k     ) - bufCurrArea.at( j, k ) );
            verVal += abs( bufCurrArea.at( j    , k + 1 ) - bufCurrArea.at( j, k ) );
            dowVal += abs( bufCurrArea.at( j + 1, k )     - bufCurrArea.at( j, k + 1 ) );
            dupVal += abs( bufCurrArea.at( j + 1, k + 1 ) - bufCurrArea.at( j, k ) );
          }
        }
        if( horVal > th * verVal && sqrt( 2 ) * horVal > th * dowVal && sqrt( 2 ) * horVal > th * dupVal && ( getPartSplit( encTestmode ) == CU_HORZ_SPLIT || getPartSplit( encTestmode ) == CU_TRIH_SPLIT ) )
        {
          return false;
        }
        if( th * dupVal < sqrt( 2 ) * verVal && th * dowVal < sqrt( 2 ) * verVal && th * horVal < verVal && ( getPartSplit( encTestmode ) == CU_VERT_SPLIT || getPartSplit( encTestmode ) == CU_TRIV_SPLIT ) )
        {
          return false;
        }
      }

      if( m_pcEncCfg->getIntraPeriod() == 1 && cuWidth <= 32 && cuHeight <= 32 && bestCS && bestCS->tus.size() == 1 && bestCU && bestCU->depth == partitioner.currDepth && partitioner.currBtDepth > 1 && isLuma( partitioner.chType ) )
      {
        if( !bestCU->rootCbf )
        {
          return false;
        }
      }
    }

    if( bestCU && bestCU->skip && bestCU->mtDepth >= m_skipThreshold && !isModeSplit( cuECtx.lastTestMode ) )
    {
      return false;
    }

    int featureToSet = -1;

    switch( getPartSplit( encTestmode ) )
    {
      case CU_QUAD_SPLIT:
        {
#if ENABLE_SPLIT_PARALLELISM
          if( !cuECtx.isLevelSplitParallel )
#endif
          if( !cuECtx.get<bool>( QT_BEFORE_BT ) && bestCU )
          {
            unsigned maxBTD        = cs.pcv->getMaxBtDepth( slice, partitioner.chType );
            const CodingUnit *cuBR = bestCS->cus.back();
            unsigned height        = partitioner.currArea().lumaSize().height;

            if (bestCU && ((bestCU->btDepth == 0 && maxBTD >= ((slice.isIntra() && !slice.getSPS()->getIBCFlag()) ? 3 : 2))
              || (bestCU->btDepth == 1 && cuBR && cuBR->btDepth == 1 && maxBTD >= ((slice.isIntra() && !slice.getSPS()->getIBCFlag()) ? 4 : 3)))
              && (width <= MAX_TB_SIZEY && height <= MAX_TB_SIZEY)
              && cuECtx.get<bool>(DID_HORZ_SPLIT) && cuECtx.get<bool>(DID_VERT_SPLIT))
            {
              return false;
            }
          }
          if( m_pcEncCfg->getUseEarlyCU() && bestCS->cost != MAX_DOUBLE && bestCU && bestCU->skip)// && !(bestCU->lheight() == 128 || bestCU->lwidth() == 128)) //&& bestCS->cus.size() == 1 )
          {
            return false;
          }
          if( getFastDeltaQp() && width <= slice.getPPS()->pcv->fastDeltaQPCuMaxSize )
          {
            return false;
          }
        }
        break;
      case CU_HORZ_SPLIT:
        featureToSet = DID_HORZ_SPLIT;
        break;
      case CU_VERT_SPLIT:
        featureToSet = DID_VERT_SPLIT;
        break;
      case CU_TRIH_SPLIT:

		if (cuECtx.get<bool>(DID_HORZ_SPLIT) && bestCU && bestCU->btDepth == partitioner.currBtDepth && !bestCU->rootCbf)
        {
          return false;
        }

        if( !cuECtx.get<bool>( DO_TRIH_SPLIT ) )
        {
          return false;
        }
        break;
      case CU_TRIV_SPLIT:
        if( cuECtx.get<bool>( DID_VERT_SPLIT ) && bestCU && bestCU->btDepth == partitioner.currBtDepth && !bestCU->rootCbf )
        {
          return false;
        }

        if( !cuECtx.get<bool>( DO_TRIV_SPLIT ) )
        {
          return false;
        }
        break;
      default:
        THROW( "Only CU split modes are governed by the EncModeCtrl" );
        return false;
        break;
    }

    switch( split )
    {
      case CU_HORZ_SPLIT:
      case CU_TRIH_SPLIT:
        if( cuECtx.get<bool>( QT_BEFORE_BT ) && cuECtx.get<bool>( DID_QUAD_SPLIT ) )
        {
          if( cuECtx.get<int>( MAX_QT_SUB_DEPTH ) > partitioner.currQtDepth + 1 )
          {
            if( featureToSet >= 0 ) cuECtx.set( featureToSet, false );
            return false;
          }
        }
        break;
      case CU_VERT_SPLIT:
      case CU_TRIV_SPLIT:
        if( cuECtx.get<bool>( QT_BEFORE_BT ) && cuECtx.get<bool>( DID_QUAD_SPLIT ) )
        {
          if( cuECtx.get<int>( MAX_QT_SUB_DEPTH ) > partitioner.currQtDepth + 1 )
          {
            if( featureToSet >= 0 ) cuECtx.set( featureToSet, false );
            return false;
          }
        }
        break;
      default:
        break;
    }

    if( split == CU_QUAD_SPLIT ) cuECtx.set( DID_QUAD_SPLIT, true );
#if JVET_Q0297_MER
    if (cs.sps->getLog2ParallelMergeLevelMinus2())
    {
      const CompArea& area = partitioner.currArea().Y();
      const SizeType size = 1 << (cs.sps->getLog2ParallelMergeLevelMinus2() + 2);
      if (!cs.slice->isIntra() && (area.width > size || area.height > size))
      {
        if (area.height <= size && split == CU_HORZ_SPLIT) return false;
        if (area.width <= size && split == CU_VERT_SPLIT) return false;
        if (area.height <= 2 * size && split == CU_TRIH_SPLIT) return false;
        if (area.width <= 2 * size && split == CU_TRIV_SPLIT) return false;
      }
    }
#endif
    return true;
  }
  else
  {
    CHECK( encTestmode.type != ETM_POST_DONT_SPLIT, "Unknown mode" );
    if ((cuECtx.get<double>(BEST_NO_IMV_COST) == (MAX_DOUBLE * .5) 
#if REUSE_CU_RESULTS
		|| cuECtx.get<bool>(IS_REUSING_CU)
#endif
		) && !slice.isIntra())
    {
      unsigned idx1, idx2, idx3, idx4;
      getAreaIdx(partitioner.currArea().Y(), *slice.getPPS()->pcv, idx1, idx2, idx3, idx4);
      if (g_isReusedUniMVsFilled[idx1][idx2][idx3][idx4])
      {
        m_pcInterSearch->insertUniMvCands(partitioner.currArea().Y(), g_reusedUniMVs[idx1][idx2][idx3][idx4]);
      }
    }
    if( !bestCS || ( bestCS && isModeSplit( bestMode ) ) )
    {
      return false;
    }
    else
    {
#if REUSE_CU_RESULTS
      setFromCs( *bestCS, partitioner );

#endif
      if( partitioner.modeType == MODE_TYPE_INTRA && partitioner.chType == CHANNEL_TYPE_LUMA )
      {
        return false; //not set best coding mode for intra coding pass
      }
      // assume the non-split modes are done and set the marks for the best found mode
      if( bestCS && bestCU )
      {
        if( CU::isInter( *bestCU ) )
        {
          relatedCU.isInter   = true;
          relatedCU.isSkip   |= bestCU->skip;
          relatedCU.isMMVDSkip |= bestCU->mmvdSkip;
          relatedCU.BcwIdx    = bestCU->BcwIdx;
          if (bestCU->slice->getSPS()->getUseColorTrans())
          {
            if (m_pcEncCfg->getRGBFormatFlag())
            {
              if (bestCU->colorTransform && bestCU->rootCbf)
              {
                relatedCU.selectColorSpaceOption = 1;
              }
              else
              {
                relatedCU.selectColorSpaceOption = 2;
              }
            }
            else
            {
              if (!bestCU->colorTransform || !bestCU->rootCbf)
              {
                relatedCU.selectColorSpaceOption = 1;
              }
              else
              {
                relatedCU.selectColorSpaceOption = 2;
              }
            }
          }
        }
        else if (CU::isIBC(*bestCU))
        {
          relatedCU.isIBC = true;
          relatedCU.isSkip |= bestCU->skip;
          if (bestCU->slice->getSPS()->getUseColorTrans())
          {
            if (m_pcEncCfg->getRGBFormatFlag())
            {
              if (bestCU->colorTransform && bestCU->rootCbf)
              {
                relatedCU.selectColorSpaceOption = 1;
              }
              else
              {
                relatedCU.selectColorSpaceOption = 2;
              }
            }
            else
            {
              if (!bestCU->colorTransform || !bestCU->rootCbf)
              {
                relatedCU.selectColorSpaceOption = 1;
              }
              else
              {
                relatedCU.selectColorSpaceOption = 2;
              }
            }
          }
        }
        else if( CU::isIntra( *bestCU ) )
        {
          relatedCU.isIntra   = true;
          if ( m_pcEncCfg->getUseFastISP() && cuECtx.ispWasTested && ( !relatedCU.relatedCuIsValid || bestCS->cost < relatedCU.bestCost ) )
          {
            // Compact data
            int bit0 = true;
            int bit1 = cuECtx.ispMode == NOT_INTRA_SUBPARTITIONS ? 1 : 0;
            int bit2 = cuECtx.ispMode == VER_INTRA_SUBPARTITIONS;
            int bit3 = cuECtx.ispLfnstIdx > 0;
            int bit4 = cuECtx.ispLfnstIdx == 2;
            int bit5 = cuECtx.mipFlag;
            int bit6 = cuECtx.bestCostIsp < cuECtx.bestNonDCT2Cost * 0.95;
            int val =
              (bit0) |
              (bit1 << 1) |
              (bit2 << 2) |
              (bit3 << 3) |
              (bit4 << 4) |
              (bit5 << 5) |
              (bit6 << 6) |
              ( cuECtx.bestPredModeDCT2 << 9 );
            relatedCU.ispPredModeVal     = val;
            relatedCU.bestDCT2NonISPCost = cuECtx.bestDCT2NonISPCost;
            relatedCU.bestCost           = bestCS->cost;
            relatedCU.bestNonDCT2Cost    = cuECtx.bestNonDCT2Cost;
            relatedCU.bestISPIntraMode   = cuECtx.bestISPIntraMode;
            relatedCU.relatedCuIsValid   = true;
          }
        }
#if ENABLE_SPLIT_PARALLELISM
#if REUSE_CU_RESULTS
        BestEncInfoCache::touch(partitioner.currArea());
#endif
        CacheBlkInfoCtrl::touch(partitioner.currArea());
#endif
        cuECtx.set( IS_BEST_NOSPLIT_SKIP, bestCU->skip );
      }
    }

    return false;
  }
}

bool EncModeCtrlMTnoRQT::checkSkipOtherLfnst( const EncTestMode& encTestmode, CodingStructure*& tempCS, Partitioner& partitioner )
{
  xExtractFeatures( encTestmode, *tempCS );

  ComprCUCtx& cuECtx  = m_ComprCUCtxList.back();
  bool skipOtherLfnst = false;

  if( encTestmode.type == ETM_INTRA )
  {
    if( !cuECtx.bestCS || ( tempCS->cost >= cuECtx.bestCS->cost && cuECtx.bestCS->cus.size() == 1 && CU::isIntra( *cuECtx.bestCS->cus[ 0 ] ) )
      || ( tempCS->cost <  cuECtx.bestCS->cost && CU::isIntra( *tempCS->cus[ 0 ] ) ) )
    {
      skipOtherLfnst = !tempCS->cus[ 0 ]->rootCbf;
    }
  }

  return skipOtherLfnst;
}

bool EncModeCtrlMTnoRQT::useModeResult( const EncTestMode& encTestmode, CodingStructure*& tempCS, Partitioner& partitioner )
{
  xExtractFeatures( encTestmode, *tempCS );

  ComprCUCtx& cuECtx = m_ComprCUCtxList.back();

#if COLLECT_DATASET
      std::string nameFile = filename_arg.substr(filename_arg.find_last_of("/\\") + 1);
      std::string sTracingFile_cost = "split_cost_" + nameFile.substr(0, nameFile.find_last_of(".")) + "_QP_" + to_string(qp_arg) + ".csv";

      FILE* m_trace_file_c = fopen( sTracingFile_cost.c_str(), "a+" );

      std::string format_head = "%d;%d;%d;%d;%d;%ld;%d;%.1f;";
#endif


  if(      encTestmode.type == ETM_SPLIT_BT_H )
  {
#if COLLECT_DATASET
    if (tempCS->slice->getSliceType() != I_SLICE){
     WriteFormatted_features(m_trace_file_c, format_head.c_str(), tempCS->slice->getPOC(), partitioner.currArea().lheight(), partitioner.currArea().lwidth(), partitioner.currArea().lx(), partitioner.currArea().ly(), partitioner.getSplitSeries(), encTestmode.type, tempCS->cost);  
     WriteFormatted_features(m_trace_file_c, "\n");
    }
#endif
    cuECtx.set( BEST_HORZ_SPLIT_COST, tempCS->cost );
  }
  else if( encTestmode.type == ETM_SPLIT_BT_V )
  {
#if COLLECT_DATASET
    if (tempCS->slice->getSliceType() != I_SLICE){
     WriteFormatted_features(m_trace_file_c, format_head.c_str(), tempCS->slice->getPOC(), partitioner.currArea().lheight(), partitioner.currArea().lwidth(), partitioner.currArea().lx(), partitioner.currArea().ly(), partitioner.getSplitSeries(), encTestmode.type, tempCS->cost);  
     WriteFormatted_features(m_trace_file_c, "\n");
    }
#endif 
    cuECtx.set( BEST_VERT_SPLIT_COST, tempCS->cost );
  }
#if  FEATURE_TEST
  else if (encTestmode.type == ETM_SPLIT_QT)
  {
#if COLLECT_DATASET
    if (tempCS->slice->getSliceType() != I_SLICE){
     WriteFormatted_features(m_trace_file_c, format_head.c_str(), tempCS->slice->getPOC(), partitioner.currArea().lheight(), partitioner.currArea().lwidth(), partitioner.currArea().lx(), partitioner.currArea().ly(), partitioner.getSplitSeries(), encTestmode.type, tempCS->cost);  
     WriteFormatted_features(m_trace_file_c, "\n");
    }
#endif 
	  cuECtx.set(BEST_QT_COST, tempCS->cost);
  }
#endif
  else if( encTestmode.type == ETM_SPLIT_TT_H )
  {
#if COLLECT_DATASET
    if (tempCS->slice->getSliceType() != I_SLICE){
     WriteFormatted_features(m_trace_file_c, format_head.c_str(), tempCS->slice->getPOC(), partitioner.currArea().lheight(), partitioner.currArea().lwidth(), partitioner.currArea().lx(), partitioner.currArea().ly(), partitioner.getSplitSeries(), encTestmode.type, tempCS->cost);  
     WriteFormatted_features(m_trace_file_c, "\n");
    }
#endif 
    cuECtx.set( BEST_TRIH_SPLIT_COST, tempCS->cost );
  }
  else if( encTestmode.type == ETM_SPLIT_TT_V )
  {
#if COLLECT_DATASET
    if (tempCS->slice->getSliceType() != I_SLICE){
     WriteFormatted_features(m_trace_file_c, format_head.c_str(), tempCS->slice->getPOC(), partitioner.currArea().lheight(), partitioner.currArea().lwidth(), partitioner.currArea().lx(), partitioner.currArea().ly(), partitioner.getSplitSeries(), encTestmode.type, tempCS->cost);  
     WriteFormatted_features(m_trace_file_c, "\n");
    }
#endif 
    cuECtx.set( BEST_TRIV_SPLIT_COST, tempCS->cost );
  }
  else if( encTestmode.type == ETM_INTRA )
  {
    const CodingUnit cu = *tempCS->getCU( partitioner.chType );

    if( !cu.mtsFlag )
    {
      cuECtx.bestMtsSize2Nx2N1stPass   = tempCS->cost;
    }
    if( !cu.ispMode )
    {
      cuECtx.bestCostMtsFirstPassNoIsp = tempCS->cost;
    }
  }
#if COLLECT_DATASET
  fclose(m_trace_file_c);
#endif
  if( m_pcEncCfg->getIMV4PelFast() && m_pcEncCfg->getIMV() && encTestmode.type == ETM_INTER_ME )
  {
    int imvMode = ( encTestmode.opts & ETO_IMV ) >> ETO_IMV_SHIFT;

    if( imvMode == 1 )
    {
      if( tempCS->cost < cuECtx.get<double>( BEST_IMV_COST ) )
      {
        cuECtx.set( BEST_IMV_COST, tempCS->cost );
      }
    }
    else if( imvMode == 0 )
    {
      if( tempCS->cost < cuECtx.get<double>( BEST_NO_IMV_COST ) )
      {
        cuECtx.set( BEST_NO_IMV_COST, tempCS->cost );
      }
    }
  }

  if( encTestmode.type == ETM_SPLIT_QT )
  {
    int maxQtD = 0;
    for( const auto& cu : tempCS->cus )
    {
      maxQtD = std::max<int>( maxQtD, cu->qtDepth );
    }
    cuECtx.set( MAX_QT_SUB_DEPTH, maxQtD );
  }

  int maxMtD = tempCS->pcv->getMaxBtDepth( *tempCS->slice, partitioner.chType ) + partitioner.currImplicitBtDepth;

  if( encTestmode.type == ETM_SPLIT_BT_H )
  {
    if( tempCS->cus.size() > 2 )
    {
      int h_2   = tempCS->area.blocks[partitioner.chType].height / 2;
      int cu1_h = tempCS->cus.front()->blocks[partitioner.chType].height;
      int cu2_h = tempCS->cus.back() ->blocks[partitioner.chType].height;

      cuECtx.set( DO_TRIH_SPLIT, cu1_h < h_2 || cu2_h < h_2 || partitioner.currMtDepth + 1 == maxMtD );
    }
  }
  else if( encTestmode.type == ETM_SPLIT_BT_V )
  {
    if( tempCS->cus.size() > 2 )
    {
      int w_2   = tempCS->area.blocks[partitioner.chType].width / 2;
      int cu1_w = tempCS->cus.front()->blocks[partitioner.chType].width;
      int cu2_w = tempCS->cus.back() ->blocks[partitioner.chType].width;

      cuECtx.set( DO_TRIV_SPLIT, cu1_w < w_2 || cu2_w < w_2 || partitioner.currMtDepth + 1 == maxMtD );
    }
  }

  // for now just a simple decision based on RD-cost or choose tempCS if bestCS is not yet coded
  if( tempCS->features[ENC_FT_RD_COST] != MAX_DOUBLE && ( !cuECtx.bestCS || ( ( tempCS->features[ENC_FT_RD_COST] + ( tempCS->useDbCost ? tempCS->costDbOffset : 0 ) ) < ( cuECtx.bestCS->features[ENC_FT_RD_COST] + ( tempCS->useDbCost ? cuECtx.bestCS->costDbOffset : 0 ) ) ) ) )
  {
    cuECtx.bestCS = tempCS;
    cuECtx.bestCU = tempCS->cus[0];
    cuECtx.bestTU = cuECtx.bestCU->firstTU;

    if( isModeInter( encTestmode ) )
    {
      //Here we take the best cost of both inter modes. We are assuming only the inter modes (and all of them) have come before the intra modes!!!
      cuECtx.bestInterCost = cuECtx.bestCS->cost;
    }

    return true;
  }
  else
  {
    return false;
  }
}

#if ENABLE_SPLIT_PARALLELISM
void EncModeCtrlMTnoRQT::copyState( const EncModeCtrl& other, const UnitArea& area )
{
  const EncModeCtrlMTnoRQT* pOther = dynamic_cast<const EncModeCtrlMTnoRQT*>( &other );

  CHECK( !pOther, "Trying to copy state from a different type of controller" );

  this->EncModeCtrl        ::copyState( *pOther, area );
  this->CacheBlkInfoCtrl   ::copyState( *pOther, area );
#if REUSE_CU_RESULTS
  this->BestEncInfoCache   ::copyState( *pOther, area );
#endif
  this->SaveLoadEncInfoSbt ::copyState( *pOther );

  m_skipThreshold = pOther->m_skipThreshold;
}

int EncModeCtrlMTnoRQT::getNumParallelJobs( const CodingStructure &cs, Partitioner& partitioner ) const
{
  int numJobs = 0;

  if(      partitioner.canSplit( CU_TRIH_SPLIT, cs ) )
  {
    numJobs = 6;
  }
  else if( partitioner.canSplit( CU_TRIV_SPLIT, cs ) )
  {
    numJobs = 5;
  }
  else if( partitioner.canSplit( CU_HORZ_SPLIT, cs ) )
  {
    numJobs = 4;
  }
  else if( partitioner.canSplit( CU_VERT_SPLIT, cs ) )
  {
    numJobs = 3;
  }
  else if( partitioner.canSplit( CU_QUAD_SPLIT, cs ) )
  {
    numJobs = 2;
  }
  else if( partitioner.canSplit( CU_DONT_SPLIT, cs ) )
  {
    numJobs = 1;
  }

  CHECK( numJobs >= NUM_RESERVERD_SPLIT_JOBS, "More jobs specified than allowed" );

  return numJobs;
}

bool EncModeCtrlMTnoRQT::isParallelSplit( const CodingStructure &cs, Partitioner& partitioner ) const
{
  if( partitioner.getImplicitSplit( cs ) != CU_DONT_SPLIT || cs.picture->scheduler.getSplitJobId() != 0 ) return false;
  if( cs.pps->getUseDQP() && partitioner.currQgEnable() ) return false;
  const int numJobs = getNumParallelJobs( cs, partitioner );
  const int numPxl  = partitioner.currArea().Y().area();
  const int parlAt  = m_pcEncCfg->getNumSplitThreads() <= 3 ? 1024 : 256;
  if(  cs.slice->isIntra() && numJobs > 2 && ( numPxl == parlAt || !partitioner.canSplit( CU_QUAD_SPLIT, cs ) ) ) return true;
  if( !cs.slice->isIntra() && numJobs > 1 && ( numPxl == parlAt || !partitioner.canSplit( CU_QUAD_SPLIT, cs ) ) ) return true;
  return false;
}

bool EncModeCtrlMTnoRQT::parallelJobSelector( const EncTestMode& encTestmode, const CodingStructure &cs, Partitioner& partitioner ) const
{
  // Job descriptors
  //  - 1: all non-split modes
  //  - 2: QT-split
  //  - 3: all vertical modes but TT_V
  //  - 4: all horizontal modes but TT_H
  //  - 5: TT_V
  //  - 6: TT_H
  switch( cs.picture->scheduler.getSplitJobId() )
  {
  case 1:
    // be sure to execute post dont split
    return !isModeSplit( encTestmode );
    break;
  case 2:
    return encTestmode.type == ETM_SPLIT_QT;
    break;
  case 3:
    return encTestmode.type == ETM_SPLIT_BT_V;
    break;
  case 4:
    return encTestmode.type == ETM_SPLIT_BT_H;
    break;
  case 5:
    return encTestmode.type == ETM_SPLIT_TT_V;
    break;
  case 6:
    return encTestmode.type == ETM_SPLIT_TT_H;
    break;
  default:
    THROW( "Unknown job-ID for parallelization of EncModeCtrlMTnoRQT: " << cs.picture->scheduler.getSplitJobId() );
    break;
  }
}

#endif


