s.options.device = "jack";
s.options.postln;
s.options.numOutputBusChannels = 2;
s.options.memSize = 256000;  // Set memory size to 256000 kilobytes (256 MB)
s.boot;  // Boot the server

s.waitForBoot({
	"Server booted.".postln;

	"/home/guillaum.rey2@hevs.ch/.local/share/SuperCollider/downloaded-quarks/JSONlib".loadPaths;
	"JSONlib loaded.".postln;

	try {
		var tempFilePath, filePath, fileArray, flatArray;
		var jsonData, fileName, recordPath;
		var dominantColor, emotion, artStyle, coarseness, contrast, directionality, roughness, regularity;
		var key, keyMapping, scale, scaleNotesMidi, scaleNotesFreq, ratios, durations;
		var granularSynthVolume, reverbAmount, delayAmount, detuneAmount, envModAmount, panningRange;
		var nbChords, chordComplexity;
		var consonants, sDissonants, dissonants;
		var tonicPerfectFifth, secondPerfectFifth, thirdPerfectFifth, fourthPerfectFifth, fifthPerfectFifth, sixthPerfectFifth;
		var tonicTriad, secondTriad, thirdTriad, fourthTriad, fifthTriad, sixthTriad;
		var diminishedTriad, tonicSeventh, secondSeventh, thirdSeventh, fourthSeventh, fifthSeventh, sixthSeventh, seventhSeventh;
		var detune, fx, att, rel, ftrLo, ftrHi, ftrModLo, ftrModHi, ftrResLow, ftrResHi, durLo, durHi, rate;
		var cubismDetune, cubismFx, cubismAtt, cubismRel, cubismFtrLo, cubismFtrHi, cubismFtrModLo, cubismFtrModHi, cubismFtrResLow, cubismFtrResHi, cubismDurLo, cubismDurHi;
		var abstractDetune, abstractFx, abstractAtt, abstractRel, abstractFtrLo, abstractFtrHi, abstractFtrModLo, abstractFtrModHi, abstractFtrResLow, abstractFtrResHi, abstractDurLo, abstractDurHi;
		var colorFieldDetune, colorFieldFx, colorFieldAtt, colorFieldRel, colorFieldFtrLo, colorFieldFtrHi, colorFieldFtrModLo, colorFieldFtrModHi, colorFieldFtrResLow, colorFieldFtrResHi, colorFieldDurLo, colorFieldDurHi;
		var realismDetune, realismFx, realismAtt, realismRel, realismFtrLo, realismFtrHi, realismFtrModLo, realismFtrModHi, realismFtrResLow, realismFtrResHi, realismDurLo, realismDurHi;
		var romanticismDetune, romanticismFx, romanticismAtt, romanticismRel, romanticismFtrLo, romanticismFtrHi, romanticismFtrModLo, romanticismFtrModHi, romanticismFtrResLow, romanticismFtrResHi, romanticismDurLo, romanticismDurHi;
		var symbolismDetune, symbolismFx, symbolismAtt, symbolismRel, symbolismFtrLo, symbolismFtrHi, symbolismFtrModLo, symbolismFtrModHi, symbolismFtrResLow, symbolismFtrResHi, symbolismDurLo, symbolismDurHi;
		var impressionismDetune, impressionismFx, impressionismAtt, impressionismRel, impressionismFtrLo, impressionismFtrHi, impressionismFtrModLo, impressionismFtrModHi, impressionismFtrResLow, impressionismFtrResHi, impressionismDurLo, impressionismDurHi;
		var expressionismDetune, expressionismFx, expressionismAtt, expressionismRel, expressionismFtrLo, expressionismFtrHi, expressionismFtrModLo, expressionismFtrModHi, expressionismFtrResLow, expressionismFtrResHi, expressionismDurLo, expressionismDurHi;
		var chords;
		var synthStyle;
		var file;

		"Starting to read temp file path.".postln;
		tempFilePath = "/home/guillaum.rey2@hevs.ch/convergence-du-visuel-et-de-l-audible-par-l-intelligence-artificielle/temp.txt";

		// Read the file into a 2D array
		fileArray = FileReader.read(tempFilePath);

		// Flatten the 2D array
		flatArray = fileArray.flat;

		// Join elements of the flattened array into a string
		filePath = flatArray.join("");
		//filePath = FileReader.read(tempFilePath);
		("Path to JSON file: " + filePath).postln;

		// EXTRACTION JSON
		"Parsing JSON file...".postln;
		jsonData = JSONlib.parseFile(filePath);
		//"JSON file parsed successfully.".postln;

		fileName = jsonData.at(\image).asString;
		recordPath = "/home/guillaum.rey2@hevs.ch/convergence-du-visuel-et-de-l-audible-par-l-intelligence-artificielle/recordings/"++fileName++".wav";

		dominantColor = jsonData.at(\dominant_color);
		emotion = jsonData.at(\emotion);
		artStyle = jsonData.at(\art_style);
		coarseness = jsonData.at(\texture).at(\coarseness);
		contrast = jsonData.at(\texture).at(\contrast);
		directionality = jsonData.at(\texture).at(\directionality);
		roughness = jsonData.at(\texture).at(\roughness);
		regularity = jsonData.at(\texture).at(\regularity);

		// KEY & SCALE
		keyMapping = (
			black: 60, brown: 61, dark_blue: 62, purple: 63,
			red: 64, orange: 65, green: 66, grey: 67,
			light_blue: 68, pink: 69, yellow: 70, white: 71
		);

		key = keyMapping[dominantColor.asSymbol];
		scale = if(emotion == "positive") { Scale.major } { Scale.minor }; // Scale definition
		scaleNotesMidi = scale.degrees + key; // Extract MIDI values from the scale
		scaleNotesFreq = scale.degrees.collect { |degree| (degree + key).midicps }; // Extract Hz values for the notes

		"Key and scale determined.".postln;

		// Consonant chords
		tonicPerfectFifth = [scaleNotesMidi[0] - 36, scaleNotesMidi[4] - 12, scaleNotesMidi[0], scaleNotesMidi[4], scaleNotesMidi[0] + 12];
		secondPerfectFifth = [scaleNotesMidi[1] - 36, scaleNotesMidi[5] - 12, scaleNotesMidi[1], scaleNotesMidi[5], scaleNotesMidi[1] + 12];
		thirdPerfectFifth = [scaleNotesMidi[2] - 36, scaleNotesMidi[6] - 12, scaleNotesMidi[2], scaleNotesMidi[6], scaleNotesMidi[2] + 12];
		fourthPerfectFifth = [scaleNotesMidi[3] - 36, scaleNotesMidi[0] - 12, scaleNotesMidi[3], scaleNotesMidi[0], scaleNotesMidi[3] + 12];
		fifthPerfectFifth = [scaleNotesMidi[4] - 36, scaleNotesMidi[1] - 12, scaleNotesMidi[4], scaleNotesMidi[1], scaleNotesMidi[4] + 12];
		sixthPerfectFifth = [scaleNotesMidi[5] - 36, scaleNotesMidi[2] - 12, scaleNotesMidi[5], scaleNotesMidi[2], scaleNotesMidi[5] + 12];
		consonants = [tonicPerfectFifth, secondPerfectFifth, thirdPerfectFifth, fourthPerfectFifth, fifthPerfectFifth, sixthPerfectFifth]; // Array of consonant chords

		// Slightly dissonant chords
		tonicTriad = [scaleNotesMidi[0] - 24, scaleNotesMidi[0], scaleNotesMidi[2], scaleNotesMidi[4], scaleNotesMidi[4] - 12];
		secondTriad = [scaleNotesMidi[1] - 24, scaleNotesMidi[1], scaleNotesMidi[3], scaleNotesMidi[5], scaleNotesMidi[5] - 12];
		fourthTriad = [scaleNotesMidi[2] - 24, scaleNotesMidi[2], scaleNotesMidi[4], scaleNotesMidi[6], scaleNotesMidi[6] - 12];
		thirdTriad = [scaleNotesMidi[3] - 24, scaleNotesMidi[3], scaleNotesMidi[5], scaleNotesMidi[0], scaleNotesMidi[0] - 12];
		fifthTriad = [scaleNotesMidi[4] - 24, scaleNotesMidi[4], scaleNotesMidi[6], scaleNotesMidi[1], scaleNotesMidi[1] - 12];
		sixthTriad = [scaleNotesMidi[5] - 24, scaleNotesMidi[5], scaleNotesMidi[0], scaleNotesMidi[2], scaleNotesMidi[2] - 12];
		sDissonants = [tonicTriad, secondTriad, thirdTriad, fourthTriad, fifthTriad, sixthTriad]; // Array of slightly dissonant chords

		// Dissonant chords
		diminishedTriad = [scaleNotesMidi[1], scaleNotesMidi[3], scaleNotesMidi[3] - 12, scaleNotesMidi[6], scaleNotesMidi[6] - 24];
		tonicSeventh = [scaleNotesMidi[0] - 24, scaleNotesMidi[0], scaleNotesMidi[2], scaleNotesMidi[4], scaleNotesMidi[6]];
		secondSeventh = [scaleNotesMidi[1] - 24, scaleNotesMidi[1], scaleNotesMidi[3], scaleNotesMidi[5], scaleNotesMidi[0]];
		thirdSeventh = [scaleNotesMidi[2] - 24, scaleNotesMidi[2], scaleNotesMidi[4], scaleNotesMidi[6], scaleNotesMidi[1]];
		fourthSeventh = [scaleNotesMidi[3] - 24, scaleNotesMidi[3], scaleNotesMidi[5], scaleNotesMidi[0], scaleNotesMidi[2]];
		fifthSeventh = [scaleNotesMidi[4] - 24, scaleNotesMidi[4], scaleNotesMidi[6], scaleNotesMidi[1], scaleNotesMidi[3]];
		sixthSeventh = [scaleNotesMidi[5] - 24, scaleNotesMidi[5], scaleNotesMidi[0], scaleNotesMidi[2], scaleNotesMidi[4]];
		seventhSeventh = [scaleNotesMidi[6] - 24, scaleNotesMidi[6], scaleNotesMidi[1], scaleNotesMidi[3], scaleNotesMidi[5]];
		dissonants = [diminishedTriad, tonicSeventh, secondSeventh, thirdSeventh, fourthSeventh, fifthSeventh, sixthSeventh, seventhSeventh]; // Array of dissonant chords

		"Chords prepared.".postln;

		// 'Regularity' defines the number of chords played
		nbChords = if (regularity <= 0.2) {
			5
		} {
			if (regularity > 0.2 && regularity <= 0.4) {
				4
			} {
				if (regularity > 0.4 && regularity <= 0.6) {
					3
				} {
					if (regularity > 0.6 && regularity <= 0.8) {
						2
					} {
						1
					}
				}
			}
		};

		// 'Roughness' defines the complexity of the chords, i.e. consonant, slightly dissonant, dissonant
		chordComplexity = if (roughness <= 0.3) {
			1
		} {
			if (roughness > 0.3 && roughness <= 0.6) {
				2
			} {
				3
			}
		};

		// Create array of array based on 'nbChords' and 'chordComplexity'
		chords = if (chordComplexity == 1) {
			Array.fill(nbChords, { consonants.choose })
		} {
			if (chordComplexity == 2) {
				Array.fill(nbChords, { sDissonants.choose }) // Choose from sDissonants
			} {
				Array.fill(nbChords, { dissonants.choose }) // Choose from dissonants
			}
		};

		"Chords array created.".postln;

		cubismDetune = 0.1;
		cubismFx = 0.1;
		cubismAtt = 0.5;
		cubismRel = 2;
		cubismFtrLo = 50;
		cubismFtrHi = 1000;
		cubismFtrModLo = 0.1;
		cubismFtrModHi = 0.2;
		cubismFtrResLow = 0.3;
		cubismFtrResHi = 0.5;
		cubismDurLo = 4;
		cubismDurHi = 7;

		abstractDetune = 0.3;
		abstractFx = 0.3;
		abstractAtt = 0.7;
		abstractRel = 3;
		abstractFtrLo = 80;
		abstractFtrHi = 3000;
		abstractFtrModLo = 0.2;
		abstractFtrModHi = 0.5;
		abstractFtrResLow = 0.2;
		abstractFtrResHi = 0.5;
		abstractDurLo = 0.5;
		abstractDurHi = 2;

		colorFieldDetune = 0.02;
		colorFieldFx = 0.5;
		colorFieldAtt = 3;
		colorFieldRel = 6;
		colorFieldFtrLo = 50;
		colorFieldFtrHi = 1000;
		colorFieldFtrModLo = 0.1;
		colorFieldFtrModHi = 0.2;
		colorFieldFtrResLow = 0.1;
		colorFieldFtrResHi = 0.2;
		colorFieldDurLo = 6;
		colorFieldDurHi = 9;

		realismDetune = 0.03;
		realismFx = 0.2;
		realismAtt = 0.5;
		realismRel = 2;
		realismFtrLo = 50;
		realismFtrHi = 1500;
		realismFtrModLo = 0.3;
		realismFtrModHi = 0.7;
		realismFtrResLow = 0.3;
		realismFtrResHi = 0.7;
		realismDurLo = 2;
		realismDurHi = 4;

		romanticismDetune = 0.1;
		romanticismFx = 0.3;
		romanticismAtt = 2;
		romanticismRel = 4;
		romanticismFtrLo = 30;
		romanticismFtrHi = 1200;
		romanticismFtrModLo = 0.1;
		romanticismFtrModHi = 0.4;
		romanticismFtrResLow = 0.1;
		romanticismFtrResHi = 0.3;
		romanticismDurLo = 3;
		romanticismDurHi = 6;

		symbolismDetune = 0.1;
		symbolismFx = 0.4;
		symbolismAtt = 3;
		symbolismRel = 5;
		symbolismFtrLo = 40;
		symbolismFtrHi = 1000;
		symbolismFtrModLo = 0.1;
		symbolismFtrModHi = 0.3;
		symbolismFtrResLow = 0.2;
		symbolismFtrResHi = 0.4;
		symbolismDurLo = 4;
		symbolismDurHi = 6;

		impressionismDetune = 0.2;
		impressionismFx = 0.4;
		impressionismAtt = 1.5;
		impressionismRel = 4;
		impressionismFtrLo = 50;
		impressionismFtrHi = 2000;
		impressionismFtrModLo = 0.2;
		impressionismFtrModHi = 0.4;
		impressionismFtrResLow = 0.1;
		impressionismFtrResHi = 0.3;
		impressionismDurLo = 0.5;
		impressionismDurHi = 1.5;

		expressionismDetune = 0.3;
		expressionismFx = 0.2;
		expressionismAtt = 0.5;
		expressionismRel = 2;
		expressionismFtrLo = 70;
		expressionismFtrHi = 2500;
		expressionismFtrModLo = 0.3;
		expressionismFtrModHi = 0.6;
		expressionismFtrResLow = 0.2;
		expressionismFtrResHi = 0.5;
		expressionismDurLo = 0.5;
		expressionismDurHi = 2.5;

		if (artStyle == "cubism") {
			detune = cubismDetune;
			fx = cubismFx;
			att = cubismAtt;
			rel = cubismRel;
			ftrLo = cubismFtrLo;
			ftrHi = cubismFtrHi;
			ftrModLo = cubismFtrModLo;
			ftrModHi = cubismFtrModHi;
			ftrResLow = cubismFtrResLow;
			ftrResHi = cubismFtrResHi;
			durLo = cubismDurLo;
			durHi = cubismDurHi;
			synthStyle = "saw";
			ratios = [1, 4 / 3, 3 / 2, 5 / 4, 7 / 5, 2, 8 / 5, 3, 9 / 5, 4];
			durations = [0.25, 0.5, 1, 1.5];
		} {
			if (artStyle == "abstract") {
				detune = abstractDetune;
				fx = abstractFx;
				att = abstractAtt;
				rel = abstractRel;
				ftrLo = abstractFtrLo;
				ftrHi = abstractFtrHi;
				ftrModLo = abstractFtrModLo;
				ftrModHi = abstractFtrModHi;
				ftrResLow = abstractFtrResLow;
				ftrResHi = abstractFtrResHi;
				durLo = abstractDurLo;
				durHi = abstractDurHi;
				synthStyle = "pulse";
				ratios = [1 / 2, 2 / 3, 1, 3 / 2, 2, 5 / 2, 3, 7 / 2, 4, 5, 6, 8];
				durations = [0.125, 0.25, 0.5, 0.75, 1];
			} {
				if (artStyle == "color_field") {
					detune = colorFieldDetune;
					fx = colorFieldFx;
					att = colorFieldAtt;
					rel = colorFieldRel;
					ftrLo = colorFieldFtrLo;
					ftrHi = colorFieldFtrHi;
					ftrModLo = colorFieldFtrModLo;
					ftrModHi = colorFieldFtrModHi;
					ftrResLow = colorFieldFtrResLow;
					ftrResHi = colorFieldFtrResHi;
					durLo = colorFieldDurLo;
					durHi = colorFieldDurHi;
					synthStyle = "sine";
					ratios = [1, 4 / 3, 5 / 3, 2, 3, 4];
					durations = [1, 1.5, 2, 3];
				} {
					if (artStyle == "realism") {
						detune = realismDetune;
						fx = realismFx;
						att = realismAtt;
						rel = realismRel;
						ftrLo = realismFtrLo;
						ftrHi = realismFtrHi;
						ftrModLo = realismFtrModLo;
						ftrModHi = realismFtrModHi;
						ftrResLow = realismFtrResLow;
						ftrResHi = realismFtrResHi;
						durLo = realismDurLo;
						durHi = realismDurHi;
						synthStyle = "saw";
						ratios = [1, 9 / 8, 5 / 4, 4 / 3, 3 / 2, 5 / 3, 2];
						durations = [0.5, 1, 1.5, 2];
					} {
						if (artStyle == "romanticism") {
							detune = romanticismDetune;
							fx = romanticismFx;
							att = romanticismAtt;
							rel = romanticismRel;
							ftrLo = romanticismFtrLo;
							ftrHi = romanticismFtrHi;
							ftrModLo = romanticismFtrModLo;
							ftrModHi = romanticismFtrModHi;
							ftrResLow = romanticismFtrResLow;
							ftrResHi = romanticismFtrResHi;
							durLo = romanticismDurLo;
							durHi = romanticismDurHi;
							synthStyle = "pulse";
							ratios = [1 / 2, 2 / 3, 3 / 4, 1, 5 / 4, 3 / 2, 2, 5 / 2];
							durations = [0.5, 1, 1.5, 2];
						} {
							if (artStyle == "symbolism") {
								detune = symbolismDetune;
								fx = symbolismFx;
								att = symbolismAtt;
								rel = symbolismRel;
								ftrLo = symbolismFtrLo;
								ftrHi = symbolismFtrHi;
								ftrModLo = symbolismFtrModLo;
								ftrModHi = symbolismFtrModHi;
								ftrResLow = symbolismFtrResLow;
								ftrResHi = symbolismFtrResHi;
								durLo = symbolismDurLo;
								durHi = symbolismDurHi;
								synthStyle = "sine";
								ratios = [1 / 2, 5 / 8, 3 / 4, 1, 5 / 4, 7 / 5, 2, 5 / 2, 3];
								durations = [0.5, 1, 1.5, 2, 2.5];
							} {
								if (artStyle == "impressionism") {
									detune = impressionismDetune;
									fx = impressionismFx;
									att = impressionismAtt;
									rel = impressionismRel;
									ftrLo = impressionismFtrLo;
									ftrHi = impressionismFtrHi;
									ftrModLo = impressionismFtrModLo;
									ftrModHi = impressionismFtrModHi;
									ftrResLow = impressionismFtrResLow;
									ftrResHi = impressionismFtrResHi;
									durLo = impressionismDurLo;
									durHi = impressionismDurHi;
									synthStyle = "pulse";
									ratios = [1 / 2, 4 / 3, 3 / 2, 5 / 4, 1, 2, 5 / 3, 3, 4];
									durations = [0.25, 0.5, 1, 1.5];
								} {
									if (artStyle == "expressionism") {
										detune = expressionismDetune;
										fx = expressionismFx;
										att = expressionismAtt;
										rel = expressionismRel;
										ftrLo = expressionismFtrLo;
										ftrHi = expressionismFtrHi;
										ftrModLo = expressionismFtrModLo;
										ftrModHi = expressionismFtrModHi;
										ftrResLow = expressionismFtrResLow;
										ftrResHi = expressionismFtrResHi;
										durLo = expressionismDurLo;
										durHi = expressionismDurHi;
										synthStyle = "saw";
										ratios = [1 / 2, 3 / 5, 4 / 3, 5 / 4, 1, 9 / 5, 2, 3, 5];
										durations = [0.25, 0.5, 0.75, 1];
									}
								}
							}
						}
					}
				}
			}
		};

		// EFFECTS AND MODULATION AMOUNT
		granularSynthVolume = 0.008 + 2*(coarseness / 1000);
		reverbAmount = (1 - contrast) + fx;
		delayAmount = contrast/2 ;
		detuneAmount = detune + (roughness / 4);
		envModAmount = regularity * 2;
		panningRange = 1 - directionality;
		rate = ((1 - coarseness) * 4) + 1 ;

		"Effect amounts determined.".postln;

		// DEFINITIONS SYNTHS
		b = Buffer.readChannel(s, "/home/guillaum.rey2@hevs.ch/granularsounds/granularfire.wav", channels: [0]);

		SynthDef(\reverb, {
			arg in, predelay = 0.1, revtime = 1.8, lpf = 4500, mix = 0.5, amp = 1, out = 0;
			var dry, wet, temp, sig;
			dry = In.ar(in, 2);
			temp = In.ar(in, 2);
			wet = 0;
			temp = DelayN.ar(temp, 0.2, predelay);
			16.do {
				temp = AllpassN.ar(temp, 0.05, { Rand(0.001, 0.05) }!2, revtime);
				temp = LPF.ar(temp, lpf);
				wet = wet + temp;
			};
			sig = XFade2.ar(dry, wet, mix * 2 - 1, amp);
			Out.ar(out, sig);
		}).add;

		SynthDef(\delay, {
			arg in, predelay = 0.5, delayTime = 1.0, feedback = 0.7, lpf = 1500, mix = 0.5, amp = 1, out = 0;
			var dry, wet, temp, sig;
			dry = In.ar(in, 2);
			temp = In.ar(in, 2);
			wet = 0;
			temp = DelayN.ar(temp, 0.2, predelay);
			16.do {
				temp = DelayC.ar(temp, delayTime, delayTime);
				temp = temp * feedback;
				temp = LPF.ar(temp, lpf);
				wet = wet + temp;
			};
			sig = XFade2.ar(dry, wet, mix * 2 - 1, amp);
			//sig = Limiter.ar(sig, 0.1, 0.01);
			Out.ar(out, sig);
		}).add;

		SynthDef(\saw, {
			arg atk = 2, sus = 0, rel = 3, c1 = 1, c2 = (-1),
			freq = 500, detune = 0.2, pan = 0, cfhzmin = 0.1, cfhzmax = 0.3,
			cfmin = 100, cfmax = 2000, rqmin = 0.1, rqmax = 0.2,
			lsf = 200, ldb = 0, amp = 1, out = 0, reverb, delay;
			var sig, env;
			env = EnvGen.kr(Env([0, 1, 1, 0], [atk, sus, rel], [c1, 0, c2]), doneAction: 2);
			sig = Saw.ar(freq * { LFNoise1.kr(0.5, detune).midiratio }!2);
			sig = BPF.ar(
				sig,
				{ LFNoise1.kr(LFNoise1.kr(4).exprange(cfhzmin, cfhzmax)).exprange(cfmin, cfmax) }!2,
				{ LFNoise1.kr(0.1).exprange(rqmin, rqmax) }!2
			);
			sig = BLowShelf.ar(sig, lsf, 0.5, ldb);
			sig = Balance2.ar(sig[0], sig[1], pan);
			sig = sig * env * amp;
			Out.ar(reverb, sig); // Output to reverb bus
			Out.ar(delay, sig);  // Output to delay bus
		}).add;

		SynthDef(\pulse, {
			arg atk = 0.1, sus = 0.5, rel = 0.3, c1 = 1, c2 = (-1),
			freq = 800, detune = 0.3, pan = 0, cfhzmin = 0.2, cfhzmax = 0.5,
			cfmin = 800, cfmax = 4000, rqmin = 0.2, rqmax = 0.5,
			lsf = 300, ldb = 6, amp = 1, out = 0, reverb, delay;
			var sig, env;
			env = EnvGen.kr(Env([0, 1, 1, 0], [atk, sus, rel], [c1, 0, c2]), doneAction: 2);
			sig = Pulse.ar(freq * { LFNoise1.kr(0.5, detune).midiratio }!2);
			sig = BPF.ar(
				sig,
				{ LFNoise1.kr(LFNoise1.kr(4).exprange(cfhzmin, cfhzmax)).exprange(cfmin, cfmax) }!2,
				{ LFNoise1.kr(0.2).exprange(rqmin, rqmax) }!2
			);
			sig = BLowShelf.ar(sig, lsf, 0.5, ldb);
			sig = Balance2.ar(sig[0], sig[1], pan);
			sig = sig * env * amp;
			Out.ar(reverb, sig);
			Out.ar(delay, sig);
		}).add;

		SynthDef(\sine, {
			arg atk = 1, sus = 2, rel = 3, c1 = 1, c2 = (-1),
			freq = 400, detune = 0.05, pan = 0, cfhzmin = 0.1, cfhzmax = 0.3,
			cfmin = 400, cfmax = 1000, rqmin = 0.2, rqmax = 0.4,
			lsf = 150, ldb = 6, amp = 0.8, out = 0, reverb, delay;
			var sig, env;
			env = EnvGen.kr(Env([0, 1, 1, 0], [atk, sus, rel], [c1, 0, c2]), doneAction: 2);
			sig = SinOsc.ar(freq * { LFNoise1.kr(0.5, detune).midiratio }!2);
			sig = BPF.ar(
				sig,
				{ LFNoise1.kr(LFNoise1.kr(4).exprange(cfhzmin, cfhzmax)).exprange(cfmin, cfmax) }!2,
				{ LFNoise1.kr(0.2).exprange(rqmin, rqmax) }!2
			);
			sig = BLowShelf.ar(sig, lsf, 0.5, ldb);
			sig = Balance2.ar(sig[0], sig[1], pan);
			sig = sig * env * amp;
			Out.ar(reverb, sig);
			Out.ar(delay, sig);
		}).add;

		SynthDef(\granular, {
			arg bufnum, amp = 0.5, out = 0, reverb, dur = 1, rate = 5;
			var sig, env;
			sig = GrainBuf.ar(
				2,
				Impulse.ar(50),
				dur,
				bufnum,
				rate,
				(
					Phasor.ar(0, 1 * BufRateScale.ir(bufnum), 0, BufSamples.ir(bufnum) - 1) + LFNoise1.ar(100).bipolar(0.01 * SampleRate.ir)
				) / BufSamples.ir(bufnum),
				5,
				0,
				-1,
				512
			) * amp;
			Out.ar(reverb, sig);
		}).add;

		"SynthDefs added.".postln;

		~reverbBus = Bus.audio(s, 2);
		~reverbSynth = Synth(\reverb, [\in, ~reverbBus]);
		~delayBus = Bus.audio(s, 2);
		~delaySynth = Synth(\delay, [\in, ~delayBus]);

		~chords = Pbind(
			\instrument, synthStyle,
			\dur, Pwhite(durLo, durHi, inf), // durée entre deux events
			\midinote, Pxrand(chords, inf),
			\detune, Pexprand(0.05, detuneAmount, inf),
			\cfmin, ftrLo,
			\cfmax, ftrHi,
			\cfhzmin, ftrModLo,
			\cfhzmax, ftrModHi,
			\rqmin, Pexprand(0.01, 0.15, inf),
			\atk, Pwhite(att, att+envModAmount, inf),
			\rel, Pwhite(rel, rel+envModAmount, inf),
			\ldb, 6,
			\amp, 0.1,
			\out, 0,
			\reverb, ~reverbBus
		).play;

		~melody = Pbind(
			\instrument, synthStyle,
			\dur, Prand(durations, inf),
			\freq, Prand(ratios, inf),
			\detune, Pwhite(0, detuneAmount, inf),
			\rqmin, 0.005,
			\rqmax, 0.008,
			\cfmin, Prand(scaleNotesMidi, inf).midicps * Prand([0.5, 1, 2, 4], inf),
			\cfmax, Pkey(\cfmin) * Pwhite(1.008, 1.025, inf),
			\atk, 0.1,
			\sus, 0.4,
			\rel, rel/3,
			\amp, 0.9,
			\pan, Pwhite(directionality*(-1), directionality, inf),
			\out, 0,
			\reverb, ~reverbBus,
			\delay, ~delayBus
		).play;

		~granular = Pbind(
			\instrument, \granular,
			\bufnum, b.bufnum,
			\dur, 0.5,
			\rate, rate,
			\reverb, ~reverbBus,
			\amp, granularSynthVolume
		).play;

		~reverbSynth.set(\mix, reverbAmount);
		~delaySynth.set(\mix, delayAmount);

		"Synths started.".postln;

		// RECORDING
		s.record(recordPath);
		"Recording started.".postln;

		// Stop synths after 20 seconds
		Routine({
			50.wait;
			~chords.stop;
			~melody.stop;
			~granular.stop;
			b.free;
			"Stopped synths.".postln;
		}).play(SystemClock);

		// Stop recording and free server after 30 seconds
		Routine({
			60.wait;
			s.stopRecording;
			{
				file = File(recordPath ++ ".done", "w");
				file.openWrite;
				file.write("done");
				file.close;
				"Done file created.".postln;
			}.defer;
			s.freeAll;
			"Recording stopped and server freed.".postln;
			s.quit;
			"Quit server".postln;
		}).play(SystemClock);
	}{
		|error|
		error.postln;
		"Error in main process.".postln;
	}
});
