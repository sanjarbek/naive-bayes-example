using System;

namespace NaiveBayes
{
	class NaiveBayesProgram
	{
		static void Main (string[] args)
		{
			try {
				string filePath = "../../data/car.data";
				
				Console.WriteLine ("\nBegin Naive Bayes Classification demo\n");
				Console.WriteLine ("Demo will classify class based on buying, maint, doors, persons, lug_boot, safety, price, class\n");

				string[] attributes = new string[] { 
					"buying", 
					"maint", 
					"doors", 
					"persons",
					"lug_boot",
					"safety",
					"price",
					"class" };
				
				string[][] attributeValues = new string[attributes.Length][];  // could scan values from raw data
				attributeValues [0] = new string[] { "vhigh", "high", "med", "low" };
				attributeValues [1] = new string[] { "vhigh", "high", "med", "low" };
				attributeValues [2] = new string[] { "2", "3", "4", "5more" };
				attributeValues [3] = new string[] { "2", "4", "more" };
				attributeValues [4] = new string[] { "small", "med", "big" };
				attributeValues [5] = new string[] { "low", "med", "high" };
				attributeValues [6] = new string[] {"cheap", "med", "high", "lux"};
				// class
				attributeValues [7] = new string[] { "unacc", "acc", "good", "vgood" };
				
				double[][] numericAttributeBorders = new double[1][];     // there may be several numeric variables
				numericAttributeBorders [0] = new double[] { 1000, 10000, 50000 };

				string[] data = ReadDataFromFile (filePath);
				Console.WriteLine ("Reading " + data.Length.ToString () + " lines of data\n");

				Console.WriteLine ("First 5 lines of training data are:\n");
				for (int i = 0; i < 5; ++i)
					Console.WriteLine (data [i]);
				Console.WriteLine ("\n");
				
				Console.WriteLine ("Converting numeric height data to categorical data on 1000, 10000, 50000\n");
			
				string[] binnedData = BinData (data, attributeValues, numericAttributeBorders);
				
				Console.WriteLine ("First 5 lines of binned training data are:\n");
				for (int i = 0; i < 5; ++i)
					Console.WriteLine (binnedData [i]);
				Console.WriteLine ("\n");
				
				Console.WriteLine ("Scanning data to compute joint and dependent counts\n"); 
				int[][][] jointCounts = MakeJointCounts (binnedData, attributes, attributeValues);
				
				int[] dependentCounts = MakeDependentCounts (jointCounts, 4);
				
				Console.WriteLine ("Total unacc = " + dependentCounts [0]);
				Console.WriteLine ("Total acc = " + dependentCounts [1]);
				Console.WriteLine ("Total good = " + dependentCounts [2]);
				Console.WriteLine ("Total vgood = " + dependentCounts [3]);
				Console.WriteLine ("");

				ShowJointCounts (jointCounts, attributeValues);
				
				while (true) {
					Console.WriteLine ("Do you want to enter test data? (Yes/No) : ");
					string response = Console.ReadLine ();
					if (!(response == "Yes" || response == "yes" || response == "y" || response == "Y"))
						break;
					
					// test data					
					string buying;
					string maint;
					string doors;
					string person;
					string lug_boot;
					string safety;
					string price;
				
					Console.WriteLine ("Enter value of buying attribute (vhigh, high, med, low): ");
					buying = Console.ReadLine ().Trim ();
					Console.WriteLine ("Enter value of maint attribute (vhigh, high, med, low): ");
					maint = Console.ReadLine ().Trim ();
					Console.WriteLine ("Enter value of doors attribute (2, 3, 4, 5more): ");
					doors = Console.ReadLine ().Trim ();
					Console.WriteLine ("Enter value of person attribute (2, 4, more): ");
					person = Console.ReadLine ().Trim ();
					Console.WriteLine ("Enter value of lug_boot attribute (small, med, big): ");
					lug_boot = Console.ReadLine ().Trim ();
					Console.WriteLine ("Enter value of safety attribute (low, med, high): ");
					safety = Console.ReadLine ().Trim ();
					Console.WriteLine ("Enter value of safety attribute (cheap, med, high, lux): ");
					price = Console.ReadLine ().Trim ();
					
					
					bool withLaplacian = true;  // prevent joint counts with 0

					Console.WriteLine ("Using Naive Bayes " + (withLaplacian ? "with" : "without") + " Laplacian smoothing to classify when:");
					Console.WriteLine (" buying = " + buying);
					Console.WriteLine (" maint = " + maint);
					Console.WriteLine (" doors = " + doors);
					Console.WriteLine (" person = " + person);
					Console.WriteLine (" lug_boot = " + lug_boot);
					Console.WriteLine (" safety = " + safety);
					Console.WriteLine (" price = " + price);
					Console.WriteLine ("");

					int c = Classify (buying, maint, doors, person, lug_boot, safety, price, jointCounts, dependentCounts, withLaplacian, 3); 
					switch (c) {
					case 0:
						Console.WriteLine ("\nData case is most likely unacc");
						break;
					case 1:
						Console.WriteLine ("\nData case is most likely acc");
						break;
					case 2:
						Console.WriteLine ("\nData case is most likely good");
						break;
					case 3:
						Console.WriteLine ("\nData case is most likely vgood");
						break;
					}
				
					Console.WriteLine ("\nEnd testing\n");
					Console.ReadLine ();
				}
				
			} catch (Exception ex) {
				Console.WriteLine (ex.Message);
				Console.ReadLine ();
			}
		} // Main
		
		static string[] ReadDataFromFile (string filePath)
		{
			string[] result = System.IO.File.ReadAllLines (filePath);
			return result;			
		}

		static int[][][] MakeJointCounts (string[] data, string[] attributes, string[][] attributeValues)
		{
			// assumes data is buying, maint, doors, person, lug_boot, safety
			// result[][][] -> [attribute][att value][class]
			// ex: result[0][3][1] is the count of (buying) (med) (acc), i.e., the count of med AND acc

			int[][][] jointCounts = new int[attributes.Length - 1][][]; // note the -1 (no class)

			jointCounts [0] = new int[4][]; // 4 buying
			jointCounts [1] = new int[4][]; // 4 maint
			jointCounts [2] = new int[4][]; // 4 doors
			jointCounts [3] = new int[3][]; // 3 person
			jointCounts [4] = new int[3][]; // 3 lug_boot
			jointCounts [5] = new int[3][]; // 3 safety
			jointCounts [6] = new int[4][]; // 4 price

			jointCounts [0] [0] = new int[4]; // 4 class for vhigh
			jointCounts [0] [1] = new int[4]; // high
			jointCounts [0] [2] = new int[4]; // med
			jointCounts [0] [3] = new int[4]; // low
			
			jointCounts [1] [0] = new int[4]; // 4 class for vhigh
			jointCounts [1] [1] = new int[4]; // high
			jointCounts [1] [2] = new int[4]; // med
			jointCounts [1] [3] = new int[4]; // low
			
			jointCounts [2] [0] = new int[4]; // 4 class for 2
			jointCounts [2] [1] = new int[4]; // 3
			jointCounts [2] [2] = new int[4]; // 4
			jointCounts [2] [3] = new int[4]; // 5more
			
			jointCounts [3] [0] = new int[4]; // 4 class for 2
			jointCounts [3] [1] = new int[4]; // 4
			jointCounts [3] [2] = new int[4]; // more
			
			jointCounts [4] [0] = new int[4]; // 4 class for small
			jointCounts [4] [1] = new int[4]; // med
			jointCounts [4] [2] = new int[4]; // high
			
			jointCounts [5] [0] = new int[4]; // 4 class for low
			jointCounts [5] [1] = new int[4]; // med
			jointCounts [5] [2] = new int[4]; // high
			
			jointCounts [6] [0] = new int[4]; // 4 class for cheap
			jointCounts [6] [1] = new int[4]; // med
			jointCounts [6] [2] = new int[4]; // high
			jointCounts [6] [3] = new int[4]; // lux

			for (int i = 0; i < data.Length; ++i) {
				string[] tokens = data [i].Split (',');

				int buyingIndex = AttributeValueToIndex (0, tokens [0].ToString ().Trim ());
				int maintIndex = AttributeValueToIndex (1, tokens [1]);
				int doorsIndex = AttributeValueToIndex (2, tokens [2]);
				int personIndex = AttributeValueToIndex (3, tokens [3]);
				int lug_bootIndex = AttributeValueToIndex (4, tokens [4]);
				int safetyIndex = AttributeValueToIndex (5, tokens [5]);
				int priceIndex = AttributeValueToIndex (6, tokens [6]);
				int classIndex = AttributeValueToIndex (7, tokens [7]);

				++jointCounts [0] [buyingIndex] [classIndex];  // buying and class count
				++jointCounts [1] [maintIndex] [classIndex];
				++jointCounts [2] [doorsIndex] [classIndex];
				++jointCounts [3] [personIndex] [classIndex];
				++jointCounts [4] [lug_bootIndex] [classIndex];
				++jointCounts [5] [safetyIndex] [classIndex];
				++jointCounts [6] [priceIndex] [classIndex];
			}

			return jointCounts;
		}

		static int AttributeValueToIndex (int attribute, string attributeValue)
		{
			switch (attribute) {
			case 0:
				if (attributeValue == "vhigh") 
					return 0;
				else if (attributeValue == "high")
					return 1;
				else if (attributeValue == "med")
					return 2;
				else if (attributeValue == "low")
					return 3;
				break;
			case 1:
				if (attributeValue == "vhigh") 
					return 0;
				else if (attributeValue == "high")
					return 1;
				else if (attributeValue == "med")
					return 2;
				else if (attributeValue == "low")
					return 3;
				break;
			case 2:
				if (attributeValue == "2") 
					return 0;
				else if (attributeValue == "3")
					return 1;
				else if (attributeValue == "4")
					return 2;
				else if (attributeValue == "5more")
					return 3;
				break;
			case 3:
				if (attributeValue == "2") 
					return 0;
				else if (attributeValue == "4")
					return 1;
				else if (attributeValue == "more")
					return 2;
				break;
			case 4:
				if (attributeValue == "small") 
					return 0;
				else if (attributeValue == "med")
					return 1;
				else if (attributeValue == "big")
					return 2;
				break;
			case 5:
				if (attributeValue == "low") 
					return 0;
				else if (attributeValue == "med")
					return 1;
				else if (attributeValue == "high")
					return 2;
				break;
			case 6:
				if (attributeValue == "cheap") 
					return 0;
				else if (attributeValue == "med")
					return 1;
				else if (attributeValue == "high")
					return 2;
				else if (attributeValue == "lux")
					return 3;
				break;
			case 7:
				if (attributeValue == "unacc") 
					return 0;
				else if (attributeValue == "acc")
					return 1;
				else if (attributeValue == "good")
					return 2;
				else if (attributeValue == "vgood")
					return 3;
				break;
			}
			
			return -1; // error
		}

		static void ShowJointCounts (int[][][] jointCounts, string[][] attributeValues)
		{
			for (int k = 0; k < 4; ++k) {
				for (int i = 0; i < jointCounts.Length; ++i)
					for (int j = 0; j < jointCounts[i].Length; ++j)
						Console.WriteLine (attributeValues [i] [j].PadRight (15) + "& " 
			                   + attributeValues [7] [k].PadRight (6) + " = " + jointCounts [i] [j] [k]);
				Console.WriteLine (""); // separate classes
			}
		}

		static int[] MakeDependentCounts (int[][][] jointCounts, int numDependents)
		{
			int[] result = new int[numDependents];
			for (int k = 0; k < numDependents; ++k)  // unacc then acc then good then vgood
				for (int j = 0; j < jointCounts[0].Length; ++j) // scanning attribute 0 = buying. could use any attribute
					result [k] += jointCounts [0] [j] [k];

			return result;
		}

		static int Classify (string buying, string maint, string doors, string person, string lug_boot, string safety, string price, int[][][] jointCounts, int[] dependentCounts, bool withSmoothing, int xClasses)
		{
			double partProbUnacc = PartialProbability ("unacc", buying, maint, doors, person, lug_boot, safety, price, jointCounts, dependentCounts, withSmoothing, xClasses);
			double partProbAcc = PartialProbability ("acc", buying, maint, doors, person, lug_boot, safety, price, jointCounts, dependentCounts, withSmoothing, xClasses);
			double partProbGood = PartialProbability ("good", buying, maint, doors, person, lug_boot, safety, price, jointCounts, dependentCounts, withSmoothing, xClasses);
			double partProbVgood = PartialProbability ("vgood", buying, maint, doors, person, lug_boot, safety, price, jointCounts, dependentCounts, withSmoothing, xClasses);
			
			double evidence = partProbUnacc + partProbAcc + partProbGood + partProbVgood;
			double probUnacc = partProbUnacc / evidence;
			double probAcc = partProbAcc / evidence;
			double probGood = partProbGood / evidence;
			double probVgood = partProbVgood / evidence;

			//Console.WriteLine("Partial prob of unacc   = " + partProbUnacc.ToString("F6"));
			//Console.WriteLine("Partial prob of acc = " + partProbAcc.ToString("F6"));
			//Console.WriteLine("Partial prob of good = " + partProbGood.ToString("F6"));
			//Console.WriteLine("Partial prob of vgood = " + partProbVgood.ToString("F6"));
      
			Console.WriteLine ("Probability of unacc   = " + probUnacc.ToString ("F4"));
			Console.WriteLine ("Probability of acc = " + probAcc.ToString ("F4"));
			Console.WriteLine ("Probability of good = " + probGood.ToString ("F4"));
			Console.WriteLine ("Probability of vgood = " + probVgood.ToString ("F4"));
			
			int result = 0;  //default result is unacc class
			
			// find max value between 4 prob. variables
			if (probUnacc < probAcc) {
				if (probAcc < probGood) {
					if (probGood < probVgood)
						result = 3;
					else
						result = 2;
				} else if (probAcc < probVgood) {
					result = 3;
				} else
					result = 1;
			} else if (probUnacc < probGood) {
				if (probGood < probVgood)
					result = 3;
				else
					result = 2;
			} else if (probUnacc < probVgood) {
				result = 3;
			} else {
				result = 0;
			}
			
			return result;
		}
		
		static string[] BinData (string[] data, string[][] attributeValues, double[][] numericAttributeBorders)
		{
			// convert numeric height to "short", "medium", or "tall". assumes data is occupation,dominance,height,sex
			string[] result = new string[data.Length];
			string[] tokens;
			double priceAsDouble;
			string priceAsBinnedString;

			for (int i = 0; i < data.Length; ++i) {
				tokens = data [i].Split (',');
				priceAsDouble = double.Parse (tokens [6]);
				if (priceAsDouble <= numericAttributeBorders [0] [0]) // cheap
					priceAsBinnedString = attributeValues [6] [0];
				else if (priceAsDouble <= numericAttributeBorders [0] [1]) // medium
					priceAsBinnedString = attributeValues [6] [1];
				else if (priceAsDouble <= numericAttributeBorders [0] [2]) // high
					priceAsBinnedString = attributeValues [6] [2];
				else
					priceAsBinnedString = attributeValues [6] [3]; 

				string s = tokens [0] + "," + tokens [1] + "," + tokens [2] + "," + tokens [3] + "," + tokens [4] + "," + tokens [5] + "," + priceAsBinnedString + "," + tokens [7];
				result [i] = s;
			}
			return result;
		}

		static double PartialProbability (string v_class, string buying, string maint, string doors, string person, string lug_boot, string safety, string price, int[][][] jointCounts, int[] dependentCounts, bool withSmoothing, int xClasses)
		{
			int classIndex = AttributeValueToIndex (7, v_class);

			int buyingIndex = AttributeValueToIndex (0, buying);
			int maintIndex = AttributeValueToIndex (1, maint);
			int doorsIndex = AttributeValueToIndex (2, doors);
			int personIndex = AttributeValueToIndex (3, person);
			int lug_bootIndex = AttributeValueToIndex (4, lug_boot);
			int safetyIndex = AttributeValueToIndex (5, safety);
			int priceIndex = AttributeValueToIndex (6, price);
			
			int totalUnacc = dependentCounts [0];
			int totalAcc = dependentCounts [1];
			int totalGood = dependentCounts [2];
			int totalVgood = dependentCounts [3];
			int totalCases = totalUnacc + totalAcc + totalGood + totalVgood;

			int totalToUse = 0;
			if (v_class == "unacc")
				totalToUse = totalUnacc;
			else if (v_class == "acc")
				totalToUse = totalAcc;
			else if (v_class == "good")
				totalToUse = totalGood;
			else if (v_class == "vgood")
				totalToUse = totalVgood;

			double p0 = (totalToUse * 1.0) / (totalCases); // prob of either unacc or acc or good or vgood
			double p1 = 0.0;
			double p2 = 0.0;
			double p3 = 0.0;
			double p4 = 0.0;
			double p5 = 0.0;
			double p6 = 0.0;
			double p7 = 0.0;

			if (withSmoothing == false) {
				p1 = (jointCounts [0] [buyingIndex] [classIndex] * 1.0) / totalToUse;  
				p2 = (jointCounts [1] [maintIndex] [classIndex] * 1.0) / totalToUse;   
				p3 = (jointCounts [2] [doorsIndex] [classIndex] * 1.0) / totalToUse;   
				p4 = (jointCounts [3] [personIndex] [classIndex] * 1.0) / totalToUse;  
				p5 = (jointCounts [4] [lug_bootIndex] [classIndex] * 1.0) / totalToUse;
				p6 = (jointCounts [5] [safetyIndex] [classIndex] * 1.0) / totalToUse;  
				p7 = (jointCounts [6] [priceIndex] [classIndex] * 1.0) / totalToUse;  
			} else if (withSmoothing == true) { // Laplacian smoothing to avoid 0-count joint probabilities
				p1 = (jointCounts [0] [buyingIndex] [classIndex] + 1) / ((totalToUse + xClasses) * 1.0);  // add 1 to count in numerator, add number x classes in denominator
				p2 = (jointCounts [1] [maintIndex] [classIndex] + 1) / ((totalToUse + xClasses) * 1.0); 
				p3 = (jointCounts [2] [doorsIndex] [classIndex] + 1) / ((totalToUse + xClasses) * 1.0);
				p4 = (jointCounts [3] [personIndex] [classIndex] + 1) / ((totalToUse + xClasses) * 1.0);
				p5 = (jointCounts [4] [lug_bootIndex] [classIndex] + 1) / ((totalToUse + xClasses) * 1.0);
				p6 = (jointCounts [5] [safetyIndex] [classIndex] + 1) / ((totalToUse + xClasses) * 1.0);
				p7 = (jointCounts [6] [priceIndex] [classIndex] + 1) / ((totalToUse + xClasses) * 1.0);
			}

			//return p0 * p1 * p2 * p3 * p4 * p5 * p6 * p7; // risky if any very small values
			return Math.Exp (Math.Log (p0) + Math.Log (p1) + Math.Log (p2) + Math.Log (p3) + Math.Log (p4) + Math.Log (p5) + Math.Log (p6) + Math.Log (p7));
		}
 
		static int AnalyzeJointCounts (int[][][] jointCounts)
		{
			// check for any joint-counts that are 0 which could blow up Naive Bayes
			int zeroCounts = 0;

			for (int i = 0; i < jointCounts.Length; ++i) // attribute
				for (int j = 0; j < jointCounts[i].Length; ++j) // value
					for (int k = 0; k < jointCounts[i][j].Length; ++k) // sex
						if (jointCounts [i] [j] [k] == 0)
							++zeroCounts;
			return zeroCounts;
		}

	} // class NaiveBayesProgram
} // namespace
