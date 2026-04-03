#!/usr/bin/env zsh

emulate -L zsh
setopt extendedglob

usage() {
  print -u2 "Usage: ${0:t} pane_count window_id width height"
}

apply_custom_layout() {
  local pane_total=$1
  local window_id=$2
  local width=$3
  local height=$4
  local rows=1
  local cols
  local layout_body layout

  while (( (rows + 1) * (rows + 1) <= pane_total )); do
    (( rows++ ))
  done
  cols=$(( (pane_total + rows - 1) / rows ))

  layout_body=$(build_window_layout "$width" "$height" "$pane_total" "$cols" "$rows")
  layout="$(layout_checksum "$layout_body"),$layout_body"

  exec tmux select-layout -t "$window_id" "$layout"
}

layout_checksum() {
  local layout=$1
  local checksum=0
  local char code
  local i

  for (( i = 1; i <= ${#layout}; i++ )); do
    char=${layout[i]}
    code=$(printf '%d' "'$char")
    checksum=$(( ((checksum >> 1) + ((checksum & 1) << 15) + code) & 0xffff ))
  done

  printf '%04x' "$checksum"
}

build_row_layout() {
  local width=$1
  local height=$2
  local x=$3
  local y=$4
  local start_index=$5
  local pane_total=$6
  local base_width current_x current_width
  local pane_offset
  local -a children

  if (( pane_total == 1 )); then
    print -r -- "${width}x${height},${x},${y},${start_index}"
    return
  fi

  base_width=$(( (width - (pane_total - 1)) / pane_total ))
  current_x=$x

  for (( pane_offset = 0; pane_offset < pane_total; pane_offset++ )); do
    if (( pane_offset + 1 < pane_total )); then
      current_width=$base_width
    else
      current_width=$(( x + width - current_x ))
    fi

    children+=("${current_width}x${height},${current_x},${y},$((start_index + pane_offset))")
    (( current_x += current_width + 1 ))
  done

  print -r -- "${width}x${height},${x},${y}{${(j:,:)children}}"
}

build_window_layout() {
  local width=$1
  local height=$2
  local pane_total=$3
  local cols=$4
  local rows=$5
  local base_height current_y current_height
  local row row_panes next_index=0
  local -a children

  if (( rows == 1 )); then
    build_row_layout "$width" "$height" 0 0 0 "$pane_total"
    return
  fi

  base_height=$(( (height - (rows - 1)) / rows ))
  current_y=0

  for (( row = 0; row < rows; row++ )); do
    row_panes=$(( pane_total - row * cols ))
    if (( row_panes > cols )); then
      row_panes=$cols
    fi

    if (( row + 1 < rows )); then
      current_height=$base_height
    else
      current_height=$(( height - current_y ))
    fi

    children+=("$(build_row_layout "$width" "$current_height" 0 "$current_y" "$next_index" "$row_panes")")
    (( next_index += row_panes ))
    (( current_y += current_height + 1 ))
  done

  print -r -- "${width}x${height},0,0[${(j:,:)children}]"
}

if (( $# != 4 )); then
  usage
  exit 1
fi

apply_custom_layout "$1" "$2" "$3" "$4"
