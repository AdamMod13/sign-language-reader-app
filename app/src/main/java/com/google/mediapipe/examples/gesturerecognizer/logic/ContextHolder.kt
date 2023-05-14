package com.google.mediapipe.examples.gesturerecognizer.logic

import android.util.Log
import com.google.mediapipe.examples.gesturerecognizer.logic.model.GestureWrapper
import com.google.mediapipe.examples.gesturerecognizer.logic.recognizer.impl.DownMovementRecognizer
import com.google.mediapipe.examples.gesturerecognizer.logic.recognizer.impl.LeftMovementRecognizer
import com.google.mediapipe.examples.gesturerecognizer.logic.recognizer.impl.RightMovementRecognizer
import com.google.mediapipe.tasks.vision.gesturerecognizer.GestureRecognizerResult
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmark


object ContextHolder {
    private const val MAX_CAPACITY = 10
    private const val GESTURE_MAX_CAPACITY = 4
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
    private val gesturesArray = ArrayDeque<GestureWrapper>() // first == newest item, last == oldest item
    var currentWord: String = ""

    fun addGestureResult(result: GestureRecognizerResult) {
        Log.i(CONTEXT_HOLDER_TAG, "$gesturesArray")
        if (result.gestures().isNotEmpty()) {
            val gesture = GestureWrapper(result.gestures(), result.landmarks())
            addGesture(gesture)

            if (!doesMatchCurrent(gesture)) {
                labelsArray.clear()
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

    private fun appendLetterToCurrentWord(label: String) {
        Log.i(CONTEXT_HOLDER_TAG, "appending $label")
        labelsArray.add(label)
        if (labelsArray.size == MAX_CAPACITY) {
            val halfArraySize = labelsArray.size / 2
            val mostCommonLabel = labelsArray
                .groupingBy { it }
                .eachCount()
                .filter { it.value >= halfArraySize }
                .maxByOrNull { it.value }?.key
            labelsArray.clear()
            mostCommonLabel?.takeIf { it.isNotBlank() }?.let { nonBlankLabel ->
                currentWord += nonBlankLabel
            }
            labelsArray.clear()
        }
    }

    private fun matchDynamicGesture(label: String) {
        Log.i(CONTEXT_HOLDER_TAG, "matching dynamic gesture\nLetter = $label")
        when (label) {
            "N", /*"O", */ "C", "S" -> {
                val movement = DownMovementRecognizer().checkHandMovement(gesturesArray)
                if (movement) {
                    dynamicSignCandidatesMap[label]?.let { appendLetterToCurrentWord(it) }
                } else {
                    appendLetterToCurrentWord(label)
                }
            }
            "L" -> {
                val movement = RightMovementRecognizer().checkHandMovement(gesturesArray)
                if (movement) {
                    dynamicSignCandidatesMap[label]?.let { appendLetterToCurrentWord(it) }
                } else {
                    appendLetterToCurrentWord(label)
                }
            }
            "H" -> {
                val movement = DownMovementRecognizer().checkHandMovement(gesturesArray)
                if (movement) {
                    dynamicSignCandidatesMap[label]?.let { appendLetterToCurrentWord(it) }
                }
            }
            "SZ" -> {
                val movement = LeftMovementRecognizer().checkHandMovement(gesturesArray)
                if (movement) {
                    dynamicSignCandidatesMap[label]?.let { appendLetterToCurrentWord(it) }
                }
            }
            "K" -> {
                val rightMovementThumb = RightMovementRecognizer().checkHandMovement(gesturesArray, HandLandmark.THUMB_TIP)
                val rightMovementMiddle = RightMovementRecognizer().checkHandMovement(gesturesArray, HandLandmark.MIDDLE_FINGER_TIP)
                if (rightMovementThumb && rightMovementMiddle) {
                    dynamicSignCandidatesMap[label]?.let { appendLetterToCurrentWord(it) }
                }
            }
            "CZ" -> {
                val downMovement = DownMovementRecognizer().checkHandMovement(gesturesArray)
                val leftMovement = LeftMovementRecognizer().checkHandMovement(gesturesArray)
                if (downMovement && leftMovement) {
                    dynamicSignCandidatesMap[label]?.let { appendLetterToCurrentWord("CZ") }
                }
                if (downMovement && !leftMovement) {
                    dynamicSignCandidatesMap[label]?.let { appendLetterToCurrentWord(it) }
                }
                else {
                    Log.i(CONTEXT_HOLDER_TAG, "Not moving down or left 😥")
                }
            }
            else -> {
                gesturesArray.clear()
                appendLetterToCurrentWord(label)
            }
        }
    }

    private fun doesMatchCurrent(candidate: GestureWrapper): Boolean {
        val count = gesturesArray.count { it.getCategory() == candidate.getCategory() }
        return count > gesturesArray.size - (GESTURE_MAX_CAPACITY / 2)
    }

    private fun isDynamic(label: String): Boolean {
        return label in dynamicSignCandidatesMap
    }

    private fun addGesture(gesture: GestureWrapper) {
        gesturesArray.addLast(gesture)
        if (gesturesArray.size > GESTURE_MAX_CAPACITY) {
            gesturesArray.removeFirst()
        }
        if (gesturesArray.none { it.getCategory() == gesture.getCategory() }) {
            gesturesArray.clear()
        }
    }
}