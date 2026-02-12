#!/bin/bash
# Google Cloud 인증 스크립트
# 매일 한 번 실행하면 됩니다.
#
# 사용법:
#   ./auth_gcloud.sh

echo "============================================"
echo "Google Cloud 인증"
echo "============================================"

gcloud auth application-default login

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================"
    echo "인증 완료!"
    echo "============================================"
    echo ""
    echo "이제 테스트를 실행할 수 있습니다."
else
    echo ""
    echo "인증 실패. 다시 시도해주세요."
    exit 1
fi
