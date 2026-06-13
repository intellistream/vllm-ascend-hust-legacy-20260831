#!/usr/bin/env bash
# ---------------------------------------------------------------
# File Name     : nvm-update.sh
# File Authors  : 曾奥然 <ccmywish@qq.com>
# Created On    : <2021-01-06>
# Last Modified : <2022-11-14>
#
# nvm-update:
#
#   Update your nvm and Node.js versions
#   See: https://gitee.com/RubyMetric/nvm-cn
# ---------------------------------------------------------------

(
  cd "$NVM_DIR"
  git fetch --tags origin
  git checkout `git describe --abbrev=0 --tags --match "v[0-9]*" $(git rev-list --tags --max-count=1)`
) && \. "$NVM_DIR/nvm.sh"

echo "=> nvm更新完成"
