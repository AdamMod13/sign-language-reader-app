package com.google.mediapipe.examples.gesturerecognizer.logic

import android.util.Log
import com.google.mediapipe.examples.gesturerecognizer.logic.model.GestureWrapper
import com.google.mediapipe.examples.gesturerecognizer.logic.recognizer.impl.DownMovementRecognizer
import com.google.mediapipe.examples.gesturerecognizer.logic.recognizer.impl.LeftMovementRecognizer
import com.google.mediapipe.examples.gesturerecognizer.logic.recognizer.impl.RightMovementRecognizer
import com.google.mediapipe.tasks.vision.gesturerecognizer.GestureRecognizerResult
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmark


object ContextHolder {
    private const val MAX_CAPACITY = 5
    private const val GESTURE_MAX_CAPACITY = 10
    private const val CONTEXT_HOLDER_TAG = "ContextHolder"

    private val dynamicSignCandidatesMap = mapOf(
        "A" to "Ą",
        "C" to "Ć",
        "CZ" to "CH",
        "D" to "D",
        "E" to "Ę",
        "F" to "F",
        "G" to "G",
        "H" to "H",
        "I" to "J",
        "K" to "K",
        "L" to "Ł",
        "N" to "Ń",
        "O" to "Ó",
        "R" to "RZ",
        "S" to "Ś",
        "SZ" to "SZ",
        "Z" to "Ż/Ź"
    )
    private val labelsArray = mutableListOf<String>()
    private val gesturesArray = ArrayDeque<GestureWrapper>()
    var currentWord: String = ""

    fun addGestureResult(result: GestureRecognizerResult) {
        Log.i(CONTEXT_HOLDER_TAG, "$gesturesArray")
        if (result.gestures().isNotEmpty()) {
            val gesture = GestureWrapper(result.gestures(), result.landmarks())
            addGesture(gesture)

            if (!doesMatchCurrent(gesture)) {
                labelsArray.clear()
                gesturesArray.clear()
            }

            if (!isDynamic(gesture.getCategory())) {
                Log.i(CONTEXT_HOLDER_TAG, "static sign")
                gesturesArray.clear()
                appendLetterToCurrentWord(gesture.getCategory())
            } else {
                Log.i(CONTEXT_HOLDER_TAG, "possible dynamic sign")
                if (gesturesArray.isNotEmpty()) {
                    Log.i(CONTEXT_HOLDER_TAG, "current sign matches most")
                    matchDynamicGesture(gesture.getCategory())
                }
            }
        }
    }

    fun clearGestureArray() {
        gesturesArray.clear()
    }

    private fun appendDynamicLetterToCurrentWord(label: String) {
        appendLetterToCurrentWord(label, labelsArray.size, 4)
    }

    private fun appendLetterToCurrentWord(label: String, divider: Int = 2, threshold: Int = 10) {
        Log.i(CONTEXT_HOLDER_TAG, "appending $label")
        Log.i(CONTEXT_HOLDER_TAG, "array $labelsArray")
        Log.i(CONTEXT_HOLDER_TAG, "gestures array $gesturesArray")
        labelsArray.add(label)
        if (labelsArray.size >= threshold) {
            val appearancesThreshold = labelsArray.size / divider
            val mostCommonLabel = labelsArray
                .groupingBy { it }
                .eachCount()
                .filter { it.value >= appearancesThreshold }
                .maxByOrNull { it.value }?.key
            Log.i(CONTEXT_HOLDER_TAG, "most common label: $mostCommonLabel")
            mostCommonLabel?.takeIf { it.isNotBlank() }?.let { nonBlankLabel ->
                Log.i(CONTEXT_HOLDER_TAG, "appending $nonBlankLabel ???")
                currentWord += nonBlankLabel
            }
            labelsArray.clear()
            gesturesArray.clear()
        }
    }

    private fun matchDynamicGesture(label: String) {
        Log.i(CONTEXT_HOLDER_TAG, "matching dynamic gesture\nLetter = $label")
        val operatingArray = if (gesturesArray.size > 4) {
            gesturesArray.take(4)
        } else {
            gesturesArray
        }
        when (label) {
            "N", "O", "C", "S" -> {
                Log.i(CONTEXT_HOLDER_TAG, "full array: $gesturesArray")
                Log.i(CONTEXT_HOLDER_TAG, "slice: $operatingArray")
                val movement = DownMovementRecognizer().checkHandMovement(operatingArray)
                if (movement) {
                    dynamicSignCandidatesMap[label]?.let { appendDynamicLetterToCurrentWord(it) }
                } else {
                    appendLetterToCurrentWord(label)
                }
            }
            "L" -> {
                val movement = RightMovementRecognizer().checkHandMovement(operatingArray)
                if (movement) {
                    dynamicSignCandidatesMap[label]?.let { appendDynamicLetterToCurrentWord(it) }
                } else {
                    appendLetterToCurrentWord(label)
                }
            }
            "H" -> {
                val movement = DownMovementRecognizer().checkHandMovement(operatingArray)
                if (movement) {
                    dynamicSignCandidatesMap[label]?.let { appendDynamicLetterToCurrentWord(it) }
                }
            }
            "SZ" -> {
                val movement = LeftMovementRecognizer().checkHandMovement(operatingArray)
                if (movement) {
                    dynamicSignCandidatesMap[label]?.let { appendDynamicLetterToCurrentWord(it) }
                }
            }
//            "K" -> {
//                val rightMovementThumb = RightMovementRecognizer().checkHandMovement(
//                    operatingArray,
//                    HandLandmark.THUMB_TIP
//                )
//                val rightMovementMiddle = RightMovementRecognizer().checkHandMovement(
//                    operatingArray,
//                    HandLandmark.MIDDLE_FINGER_TIP
//                )
//                if (rightMovementThumb && rightMovementMiddle) {
//                    dynamicSignCandidatesMap[label]?.let { appendDynamicLetterToCurrentWord(it) }
//                }
//            }
            "CZ" -> {
                val downMovement = DownMovementRecognizer().checkHandMovement(operatingArray)
                val leftMovement = LeftMovementRecognizer().checkHandMovement(operatingArray)
                if (downMovement && leftMovement) {
                    dynamicSignCandidatesMap[label]?.let { appendDynamicLetterToCurrentWord("CZ") }
                }
                if (downMovement && !leftMovement) {
                    dynamicSignCandidatesMap[label]?.let { appendDynamicLetterToCurrentWord(it) }
                } else {
                    Log.i(CONTEXT_HOLDER_TAG, "Not moving down or left 😥")
                }
            }
            "I" -> {
                val downMovement = DownMovementRecognizer().checkHandMovement(operatingArray)
                val leftMovement = LeftMovementRecognizer().checkHandMovement(operatingArray)
                if (downMovement && leftMovement) {
                    dynamicSignCandidatesMap[label]?.let { appendDynamicLetterToCurrentWord("J") }
                } else {
                    appendLetterToCurrentWord(label)
                }
            }
            else -> {
                appendLetterToCurrentWord(label, 3, 15)
            }
        }
    }

    private fun doesMatchCurrent(candidate: GestureWrapper): Boolean {
        val count = gesturesArray.count { it.getCategory() == candidate.getCategory() }
        return count > gesturesArray.size - (GESTURE_MAX_CAPACITY / 4)
    }

    private fun isDynamic(label: String): Boolean {
        return label in dynamicSignCandidatesMap
    }

    private fun addGesture(gesture: GestureWrapper) {
        gesturesArray.addLast(gesture)
        if (gesturesArray.size > GESTURE_MAX_CAPACITY) {
            Log.i(CONTEXT_HOLDER_TAG, "overflow popping")
            gesturesArray.removeFirst()
        }
        if (gesturesArray.none { it.getCategory() == gesture.getCategory() }) {
            gesturesArray.clear()
        }
    }
}