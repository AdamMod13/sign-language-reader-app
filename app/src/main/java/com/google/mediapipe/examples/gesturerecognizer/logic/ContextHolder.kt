package com.google.mediapipe.examples.gesturerecognizer.logic

import android.util.Log
import com.google.mediapipe.tasks.components.containers.Category
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.vision.gesturerecognizer.GestureRecognizerResult
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmark


object ContextHolder {
    private const val MAX_CAPACITY = 10
    private const val GESTURE_MAX_CAPACITY = 4
    private const val CONTEXT_HOLDER_TAG = "ContextHolder"

    private val dynamicSignCandidatesMap = mapOf(
        "A" to "Ą",
        "C" to "Ć",
        "CZ" to "CH", // te znaki to weryfikacji
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
    private val gesturesArray = ArrayDeque<Gesture>() // first == newest item, last == oldest item
    var currentWord: String = ""

    fun addGestureResult(result: GestureRecognizerResult) {
        Log.i(CONTEXT_HOLDER_TAG, "$gesturesArray")
        if (result.gestures().isNotEmpty()) {
            val gesture = Gesture(result.gestures(), result.landmarks())
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

//    private fun checkHandMovement(
//        first: List<NormalizedLandmark>, last: List<NormalizedLandmark>
//    ): Float {
//        return last[HandLandmark.WRIST].y() - first[HandLandmark.WRIST].y()
//    }

    private fun matchDynamicGesture(label: String) {
        Log.i(CONTEXT_HOLDER_TAG, "matching dynamic gesture\n Letter = $label")
        when (label) {
            "N", "O", "C" -> {
                val movement = DownMovementRecognizer().checkHandMovement(gesturesArray)
                if (movement) {
                    dynamicSignCandidatesMap[label]?.let { appendLetterToCurrentWord(it) }
                } else {
                    appendLetterToCurrentWord(label)
                }
            }
            else -> {
                gesturesArray.clear()
                appendLetterToCurrentWord(label)
            }
        }
    }

    private fun doesMatchCurrent(candidate: Gesture): Boolean {
        val count = gesturesArray.count { it.getCategory() == candidate.getCategory() }
        return count > gesturesArray.size - (GESTURE_MAX_CAPACITY / 2)
    }

    private fun isDynamic(label: String): Boolean {
        return label in dynamicSignCandidatesMap
    }

    private fun addGesture(gesture: Gesture) {
        gesturesArray.addFirst(gesture)
        if (gesturesArray.size > GESTURE_MAX_CAPACITY) {
            gesturesArray.removeLast()
        }
        if (gesturesArray.none { it.getCategory() == gesture.getCategory() }) {
            gesturesArray.clear()
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
}

data class Gesture(
    val gesture: MutableList<MutableList<Category>>,
    val landmarks: MutableList<MutableList<NormalizedLandmark>>
) {
    fun getCategory(): String {
        return gesture[0][0].categoryName()
    }

    fun getLandmarkArray(): List<NormalizedLandmark> {
        return landmarks[0]
    }
}

interface DynamicGestureRecognizer {
    fun checkHandMovement(gestureList: List<Gesture>): Boolean
}

class DownMovementRecognizer : DynamicGestureRecognizer {
    override fun checkHandMovement(gestureList: List<Gesture>): Boolean {
        return compareFirstAndLastPosition(gestureList.first(), gestureList.last()) < 0
    }

    // todo: compare sequence of gestures, not first and last
    private fun compareFirstAndLastPosition(
        first: Gesture, last: Gesture
    ): Float {
        return last.getLandmarkArray()[HandLandmark.WRIST].y() - first.getLandmarkArray()[HandLandmark.WRIST].y()
    }
}

class UpMovementRecognizer : DynamicGestureRecognizer {
    override fun checkHandMovement(gestureList: List<Gesture>): Boolean {
        val movements = mutableListOf<Float>()
        for (i in gestureList.indices - 1) {
            val movement = compareFirstAndLastPosition(gestureList[i], gestureList[i + 1])
            Log.i("checkHandMovement", "$movement between indexed $i and ${i + 1}")
            movements.add(movement)
        }
        return movements.all { it < 0 }
    }

    // todo: compare sequence of gestures, not first and last
    private fun compareFirstAndLastPosition(
        first: Gesture, last: Gesture
    ): Float {
        return last.getLandmarkArray()[HandLandmark.WRIST].y() - first.getLandmarkArray()[HandLandmark.WRIST].y()
    }
}