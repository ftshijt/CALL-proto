#!/bin/bash

workspace=$1
echo ${workspace}
. ${workspace}/path.sh || exit 1;

stage=1
nj=$3
infer_set=$2

cd ${workspace}

if [ $stage -le 1 ]; then
  echo ============================================================================
  echo "               Data Preparation                     "
  echo ============================================================================
  for datadir in ${infer_set}; do
    ${workspace}/utils/fix_data_dir.sh ${datadir}
  done
fi
# Now make MFCC plus pitch features.
# mfccdir should be some place with a largish disk where you
# want to store MFCC features.
if [ $stage -le 2 ]; then
  echo ============================================================================
  echo "                MFCC Extraction                     "
  echo ============================================================================
  mfccdir=mfcc_hires
  for datadir in ${infer_set}; do
    ${workspace}/utils/copy_data_dir.sh ${datadir} ${datadir}_hires
    ${workspace}/steps/make_mfcc.sh --mfcc-config conf/mfcc_hires.conf \
        --nj $nj ${datadir}_hires exp/make_mfcc/ ${mfccdir}
    ${workspace}/steps/compute_cmvn_stats.sh ${datadir}_hires exp/make_mfcc ${mfccdir}
    ${workspace}/utils/fix_data_dir.sh ${datadir}_hires
  done
  
  for datadir in ${infer_set}; do
    ${workspace}/steps/online/nnet2/extract_ivectors_online.sh --nj $nj \
      ${datadir}_hires ${workspace}/exp/nnet3/extractor ${datadir}_ivector || exit 1;
  done
  
fi


if [ ${stage} -le 3 ]; then
  echo ============================================================================
  echo "                Decoding                     "
  echo ============================================================================
  
  for datadir in ${infer_set}; do
   ${workspace}/steps/chain/get_phone_post.sh --nj $nj --remove-word-position-dependency true \
      --online-ivector-dir ${datadir}_ivector \
      ${workspace}/exp/chain/tree_sp/ ${workspace}/exp/chain/tdnn_1d_sp ${workspace}/data/lang_chain ${datadir}_hires ${datadir}_post
    # for jbs in $(seq 1 ${nj}); do
    #   copy-feats ark:${datadir}_post/phone_post.${jbs}.ark ark,t:${datadir}_post/phone_post.${jbs}.txt
    # done
  done
fi

echo "infer success"
