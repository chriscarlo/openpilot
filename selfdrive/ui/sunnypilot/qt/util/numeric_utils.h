/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#pragma once

#include <string>
#include <cstdlib>

namespace SunnypilotUtils {

/**
 * Safely converts a string to integer with default value fallback.
 * Returns defaultValue if string is empty, contains non-numeric characters,
 * or if conversion fails for any reason.
 */
inline int safeStringToInt(const std::string& str, int defaultValue = 0) {
  if (str.empty()) return defaultValue;
  try {
    // Check if string contains only digits and optional leading negative sign
    if (str.find_first_not_of("0123456789-") != std::string::npos) {
      return defaultValue;
    }
    return std::atoi(str.c_str());
  } catch (...) {
    return defaultValue;
  }
}

} // namespace SunnypilotUtils