classdef Homomorphic_Image_Dehaze < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        UIFigure                  matlab.ui.Figure
        GridLayout                matlab.ui.container.GridLayout
        LeftPanel                 matlab.ui.container.Panel
        Image                     matlab.ui.control.Image
        HistogramButton           matlab.ui.control.Button
        Dehazed_RGBSwitchLabel    matlab.ui.control.Label
        Dehazed_GraySwitch        matlab.ui.control.Switch
        Dehazed_GraySwitchLabel   matlab.ui.control.Label
        GI_FFTSwitch              matlab.ui.control.Switch
        GI_FFTSwitchLabel         matlab.ui.control.Label
        TabGroup                  matlab.ui.container.TabGroup
        GausHPTab                 matlab.ui.container.Tab
        CEditField                matlab.ui.control.NumericEditField
        CEditFieldLabel           matlab.ui.control.Label
        gHEditField               matlab.ui.control.NumericEditField
        gHEditFieldLabel          matlab.ui.control.Label
        gLEditField               matlab.ui.control.NumericEditField
        gLEditFieldLabel          matlab.ui.control.Label
        D0EditField               matlab.ui.control.NumericEditField
        D0EditFieldLabel          matlab.ui.control.Label
        ButterHPTab               matlab.ui.container.Tab
        NEditField                matlab.ui.control.NumericEditField
        NEditFieldLabel           matlab.ui.control.Label
        D0EditField_2             matlab.ui.control.NumericEditField
        D0EditField_2Label        matlab.ui.control.Label
        IdealHPTab                matlab.ui.container.Tab
        D0EditField_3             matlab.ui.control.NumericEditField
        D0EditField_3Label        matlab.ui.control.Label
        Adapt_EQSwitch            matlab.ui.control.Switch
        Adapt_EQSwitchLabel       matlab.ui.control.Label
        RunButton                 matlab.ui.control.Button
        I_FFTSwitch               matlab.ui.control.Switch
        I_FFTSwitchLabel          matlab.ui.control.Label
        DispSelectedFilterButton  matlab.ui.control.Button
        SelectImageButton         matlab.ui.control.Button
        Dehazed_RGBSwitch         matlab.ui.control.Switch
        RightPanel                matlab.ui.container.Panel
        UIAxes                    matlab.ui.control.UIAxes
        UIAxes_2                  matlab.ui.control.UIAxes
    end

    % Properties that correspond to apps with auto-reflow
    properties (Access = private)
        onePanelWidth = 576;
    end

    
    properties (Access = private)
        imageData = []
        I_gray =[]
        I_gray_defog = []
        H = []
        H_selected =[]
        I_defog = []
        imageJ_adapt = []
    end
    
    methods (Access = private)
        
        function Update_H(app)
                app.Dehazed_GraySwitch.Enable='on';app.Dehazed_GraySwitchLabel.Enable='on';
                app.Dehazed_RGBSwitch.Enable='on';app.Dehazed_RGBSwitchLabel.Enable='on';
                app.Adapt_EQSwitch.Enable='on';app.Adapt_EQSwitchLabel.Enable='on';
                app.GI_FFTSwitch.Enable='on';app.GI_FFTSwitchLabel.Enable='on';
                app.I_FFTSwitch.Enable='on';app.I_FFTSwitchLabel.Enable='on';
                app.DispSelectedFilterButton.Enable='on';
                app.RunButton.Enable='on';
        end
    end
    

    % Callbacks that handle component events
    methods (Access = private)

        % Button pushed function: SelectImageButton
        function SelectImageButtonPushed(app, ~)

            close all;

            [file ,path] = uigetfile({'*.jpg;*.jpeg;*.png;*.gif;*.tif';'*.*'},'File Selector');
            selectedfile = fullfile(path,file);
            
            % Check if a file was selected
            if isequal(file, 0) % No file selected
                msgbox('There is NO image selected');
            else
                % Read the image file
                app.imageData=imread(selectedfile);
                imshow(app.imageData, 'Parent', app.UIAxes);
                [height, width, ~] = size(app.imageData);
                app.UIAxes.XLim = [0 width];
                app.UIAxes.YLim = [0 height];
                app.TabGroup.Visible='on';
                app.RunButton.Enable='on';
                app.HistogramButton.Enable='on';

                app.imageData = im2double(app.imageData);
                app.I_gray = rgb2gray(app.imageData);
            end
         
        end

        % Button pushed function: RunButton
        function RunButtonPushed(app, ~)
            close all;
            %%
            [app.I_gray_defog ,If,G_I_FFT] = homomorphic_filter(app.I_gray,app.H);

            if strcmp(app.I_FFTSwitch.Value,'On')
               figure,imshow(If),title('I_{FFT}'); 
            end
            if strcmp(app.GI_FFTSwitch.Value,'On')
               figure,imshow(G_I_FFT),title('G*I_{FFT}'); 
            end

            if strcmp(app.Dehazed_GraySwitch.Value,'On')
               figure,imshow(app.I_gray_defog),title('Dehazed Gray');
            end 
            %% 
            app.I_defog = zeros(size(app.imageData));
            for i = 1:3
                app.I_defog(:,:,i) = app.imageData(:,:,i).*app.I_gray_defog;
            end
            imshow(app.I_defog, 'Parent', app.UIAxes_2);
            [height, width, ~] = size(app.I_defog);
            app.UIAxes_2.XLim = [0 width];
            app.UIAxes_2.YLim = [0 height];
            if strcmp(app.Dehazed_RGBSwitch.Value,'On')
               figure,imshow(app.I_defog,[]),title('Dehazed');
            end 
  
            %%
            if strcmp(app.Adapt_EQSwitch.Value,'On')
               LAB = rgb2lab(uint8(app.I_defog*255)); 
               L = LAB(:,:,1)/100;
               L = adapthisteq(L,'NumTiles',[8 8],'ClipLimit',0.005);
               LAB(:,:,1) = L*100;
               app.imageJ_adapt = lab2rgb(LAB);
               figure,imshow(app.imageJ_adapt),title('J_{adaptEQ}');
            end
            
        end

        % Callback function: CEditField, D0EditField, GausHPTab, GausHPTab,
        % 
        % ...and 2 other components
        function GausHPTabButtonDown(app, ~)
          gL = app.gLEditField.Value;
          gH = app.gHEditField.Value;
          C  = app.CEditField.Value;
          D0 = app.D0EditField.Value;
          app.H = gaushp(app.I_gray, gL, gH, D0, C);  
          app.Update_H()
        end

        % Callback function: ButterHPTab, ButterHPTab, D0EditField_2, 
        % ...and 1 other component
        function ButterHPTabButtonDown(app, ~)
          n  = app.NEditField.Value;
          D0 = app.D0EditField_2.Value;
          app.H = butterhp(app.I_gray, D0,n);
          app.Update_H()
        end

        % Callback function: D0EditField_3, IdealHPTab, IdealHPTab
        function IdealHPTabButtonDown(app, ~)
          D0 = app.D0EditField_3.Value;
          app.H = idealhp(app.I_gray, D0);
          app.Update_H()
        end

        % Button pushed function: DispSelectedFilterButton
        function DispSelectedFilterButtonPushed(app, ~)
            close all;
            plotTFSurface(app.H);
        end

        % Button pushed function: HistogramButton
        function HistogramButtonPushed(app, ~)
            figure;imhist(app.I_gray);title('Histogram');
        end

        % Changes arrangement of the app based on UIFigure width
        function updateAppLayout(app, ~)
            currentFigureWidth = app.UIFigure.Position(3);
            if(currentFigureWidth <= app.onePanelWidth)
                % Change to a 2x1 grid
                app.GridLayout.RowHeight = {516, 516};
                app.GridLayout.ColumnWidth = {'1x'};
                app.RightPanel.Layout.Row = 2;
                app.RightPanel.Layout.Column = 1;
            else
                % Change to a 1x2 grid
                app.GridLayout.RowHeight = {'1x'};
                app.GridLayout.ColumnWidth = {248, '1x'};
                app.RightPanel.Layout.Row = 1;
                app.RightPanel.Layout.Column = 2;
            end
        end
    end

    % Component initialization
    methods (Access = private)

        % Create UIFigure and components
        function createComponents(app)

            % Get the file path for locating images
            pathToMLAPP = fileparts(mfilename('fullpath'));

            % Create UIFigure and hide until all components are created
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.AutoResizeChildren = 'off';
            app.UIFigure.Position = [100 100 650 516];
            app.UIFigure.Name = 'MATLAB App';
            app.UIFigure.SizeChangedFcn = createCallbackFcn(app, @updateAppLayout, true);

            % Create GridLayout
            app.GridLayout = uigridlayout(app.UIFigure);
            app.GridLayout.ColumnWidth = {248, '1x'};
            app.GridLayout.RowHeight = {'1x'};
            app.GridLayout.ColumnSpacing = 0;
            app.GridLayout.RowSpacing = 0;
            app.GridLayout.Padding = [0 0 0 0];
            app.GridLayout.Scrollable = 'on';

            % Create LeftPanel
            app.LeftPanel = uipanel(app.GridLayout);
            app.LeftPanel.Layout.Row = 1;
            app.LeftPanel.Layout.Column = 1;

            % Create Dehazed_RGBSwitch
            app.Dehazed_RGBSwitch = uiswitch(app.LeftPanel, 'slider');
            app.Dehazed_RGBSwitch.Enable = 'off';
            app.Dehazed_RGBSwitch.Position = [150 211 45 20];

            % Create SelectImageButton
            app.SelectImageButton = uibutton(app.LeftPanel, 'push');
            app.SelectImageButton.ButtonPushedFcn = createCallbackFcn(app, @SelectImageButtonPushed, true);
            app.SelectImageButton.Position = [74 483 100 23];
            app.SelectImageButton.Text = 'Select Image';

            % Create DispSelectedFilterButton
            app.DispSelectedFilterButton = uibutton(app.LeftPanel, 'push');
            app.DispSelectedFilterButton.ButtonPushedFcn = createCallbackFcn(app, @DispSelectedFilterButtonPushed, true);
            app.DispSelectedFilterButton.Enable = 'off';
            app.DispSelectedFilterButton.Position = [68 323 120 22];
            app.DispSelectedFilterButton.Text = 'Disp Selected Filter';

            % Create I_FFTSwitchLabel
            app.I_FFTSwitchLabel = uilabel(app.LeftPanel);
            app.I_FFTSwitchLabel.HorizontalAlignment = 'center';
            app.I_FFTSwitchLabel.Enable = 'off';
            app.I_FFTSwitchLabel.Position = [43 293 37 22];
            app.I_FFTSwitchLabel.Text = 'I_FFT';

            % Create I_FFTSwitch
            app.I_FFTSwitch = uiswitch(app.LeftPanel, 'slider');
            app.I_FFTSwitch.Enable = 'off';
            app.I_FFTSwitch.Position = [42 267 45 20];

            % Create RunButton
            app.RunButton = uibutton(app.LeftPanel, 'push');
            app.RunButton.ButtonPushedFcn = createCallbackFcn(app, @RunButtonPushed, true);
            app.RunButton.Enable = 'off';
            app.RunButton.Position = [114 120 110 65];
            app.RunButton.Text = 'Run';

            % Create Adapt_EQSwitchLabel
            app.Adapt_EQSwitchLabel = uilabel(app.LeftPanel);
            app.Adapt_EQSwitchLabel.HorizontalAlignment = 'center';
            app.Adapt_EQSwitchLabel.Enable = 'off';
            app.Adapt_EQSwitchLabel.Position = [38 144 61 22];
            app.Adapt_EQSwitchLabel.Text = 'Adapt_EQ';

            % Create Adapt_EQSwitch
            app.Adapt_EQSwitch = uiswitch(app.LeftPanel, 'slider');
            app.Adapt_EQSwitch.Orientation = 'vertical';
            app.Adapt_EQSwitch.Enable = 'off';
            app.Adapt_EQSwitch.Position = [12 133 20 45];

            % Create TabGroup
            app.TabGroup = uitabgroup(app.LeftPanel);
            app.TabGroup.Visible = 'off';
            app.TabGroup.Position = [12 352 218 94];

            % Create GausHPTab
            app.GausHPTab = uitab(app.TabGroup);
            app.GausHPTab.SizeChangedFcn = createCallbackFcn(app, @GausHPTabButtonDown, true);
            app.GausHPTab.Title = 'GausHP';
            app.GausHPTab.ButtonDownFcn = createCallbackFcn(app, @GausHPTabButtonDown, true);

            % Create D0EditFieldLabel
            app.D0EditFieldLabel = uilabel(app.GausHPTab);
            app.D0EditFieldLabel.HorizontalAlignment = 'right';
            app.D0EditFieldLabel.Position = [12 38 25 22];
            app.D0EditFieldLabel.Text = 'D0';

            % Create D0EditField
            app.D0EditField = uieditfield(app.GausHPTab, 'numeric');
            app.D0EditField.Limits = [1 100];
            app.D0EditField.ValueChangedFcn = createCallbackFcn(app, @GausHPTabButtonDown, true);
            app.D0EditField.Position = [52 38 30 22];
            app.D0EditField.Value = 1;

            % Create gLEditFieldLabel
            app.gLEditFieldLabel = uilabel(app.GausHPTab);
            app.gLEditFieldLabel.HorizontalAlignment = 'right';
            app.gLEditFieldLabel.Position = [130 39 25 22];
            app.gLEditFieldLabel.Text = 'gL';

            % Create gLEditField
            app.gLEditField = uieditfield(app.GausHPTab, 'numeric');
            app.gLEditField.Limits = [0.09 1];
            app.gLEditField.ValueChangedFcn = createCallbackFcn(app, @GausHPTabButtonDown, true);
            app.gLEditField.Position = [170 39 30 22];
            app.gLEditField.Value = 0.1;

            % Create gHEditFieldLabel
            app.gHEditFieldLabel = uilabel(app.GausHPTab);
            app.gHEditFieldLabel.HorizontalAlignment = 'right';
            app.gHEditFieldLabel.Position = [130 6 25 22];
            app.gHEditFieldLabel.Text = 'gH';

            % Create gHEditField
            app.gHEditField = uieditfield(app.GausHPTab, 'numeric');
            app.gHEditField.Limits = [1 4];
            app.gHEditField.ValueChangedFcn = createCallbackFcn(app, @GausHPTabButtonDown, true);
            app.gHEditField.Position = [170 6 30 22];
            app.gHEditField.Value = 1.1;

            % Create CEditFieldLabel
            app.CEditFieldLabel = uilabel(app.GausHPTab);
            app.CEditFieldLabel.HorizontalAlignment = 'right';
            app.CEditFieldLabel.Position = [12 6 25 22];
            app.CEditFieldLabel.Text = 'C';

            % Create CEditField
            app.CEditField = uieditfield(app.GausHPTab, 'numeric');
            app.CEditField.Limits = [1 100];
            app.CEditField.ValueChangedFcn = createCallbackFcn(app, @GausHPTabButtonDown, true);
            app.CEditField.Position = [52 6 30 22];
            app.CEditField.Value = 2;

            % Create ButterHPTab
            app.ButterHPTab = uitab(app.TabGroup);
            app.ButterHPTab.SizeChangedFcn = createCallbackFcn(app, @ButterHPTabButtonDown, true);
            app.ButterHPTab.Title = 'ButterHP';
            app.ButterHPTab.ButtonDownFcn = createCallbackFcn(app, @ButterHPTabButtonDown, true);

            % Create D0EditField_2Label
            app.D0EditField_2Label = uilabel(app.ButterHPTab);
            app.D0EditField_2Label.HorizontalAlignment = 'right';
            app.D0EditField_2Label.Position = [74 41 25 22];
            app.D0EditField_2Label.Text = 'D0';

            % Create D0EditField_2
            app.D0EditField_2 = uieditfield(app.ButterHPTab, 'numeric');
            app.D0EditField_2.Limits = [1 100];
            app.D0EditField_2.ValueChangedFcn = createCallbackFcn(app, @ButterHPTabButtonDown, true);
            app.D0EditField_2.Position = [114 41 30 22];
            app.D0EditField_2.Value = 1;

            % Create NEditFieldLabel
            app.NEditFieldLabel = uilabel(app.ButterHPTab);
            app.NEditFieldLabel.HorizontalAlignment = 'right';
            app.NEditFieldLabel.Position = [74 11 25 22];
            app.NEditFieldLabel.Text = 'N';

            % Create NEditField
            app.NEditField = uieditfield(app.ButterHPTab, 'numeric');
            app.NEditField.Limits = [1 100];
            app.NEditField.RoundFractionalValues = 'on';
            app.NEditField.ValueChangedFcn = createCallbackFcn(app, @ButterHPTabButtonDown, true);
            app.NEditField.Position = [114 11 30 22];
            app.NEditField.Value = 2;

            % Create IdealHPTab
            app.IdealHPTab = uitab(app.TabGroup);
            app.IdealHPTab.SizeChangedFcn = createCallbackFcn(app, @IdealHPTabButtonDown, true);
            app.IdealHPTab.Title = 'IdealHP';
            app.IdealHPTab.ButtonDownFcn = createCallbackFcn(app, @IdealHPTabButtonDown, true);

            % Create D0EditField_3Label
            app.D0EditField_3Label = uilabel(app.IdealHPTab);
            app.D0EditField_3Label.HorizontalAlignment = 'right';
            app.D0EditField_3Label.Position = [70 25 25 22];
            app.D0EditField_3Label.Text = 'D0';

            % Create D0EditField_3
            app.D0EditField_3 = uieditfield(app.IdealHPTab, 'numeric');
            app.D0EditField_3.Limits = [1 100];
            app.D0EditField_3.RoundFractionalValues = 'on';
            app.D0EditField_3.ValueChangedFcn = createCallbackFcn(app, @IdealHPTabButtonDown, true);
            app.D0EditField_3.Position = [110 25 30 22];
            app.D0EditField_3.Value = 1;

            % Create GI_FFTSwitchLabel
            app.GI_FFTSwitchLabel = uilabel(app.LeftPanel);
            app.GI_FFTSwitchLabel.HorizontalAlignment = 'center';
            app.GI_FFTSwitchLabel.Enable = 'off';
            app.GI_FFTSwitchLabel.Position = [144 292 52 22];
            app.GI_FFTSwitchLabel.Text = 'G*I_FFT';

            % Create GI_FFTSwitch
            app.GI_FFTSwitch = uiswitch(app.LeftPanel, 'slider');
            app.GI_FFTSwitch.Enable = 'off';
            app.GI_FFTSwitch.Position = [150 266 45 20];

            % Create Dehazed_GraySwitchLabel
            app.Dehazed_GraySwitchLabel = uilabel(app.LeftPanel);
            app.Dehazed_GraySwitchLabel.HorizontalAlignment = 'center';
            app.Dehazed_GraySwitchLabel.Enable = 'off';
            app.Dehazed_GraySwitchLabel.Position = [19 235 86 22];
            app.Dehazed_GraySwitchLabel.Text = 'Dehazed_Gray';

            % Create Dehazed_GraySwitch
            app.Dehazed_GraySwitch = uiswitch(app.LeftPanel, 'slider');
            app.Dehazed_GraySwitch.Enable = 'off';
            app.Dehazed_GraySwitch.Position = [42 211 45 20];

            % Create Dehazed_RGBSwitchLabel
            app.Dehazed_RGBSwitchLabel = uilabel(app.LeftPanel);
            app.Dehazed_RGBSwitchLabel.HorizontalAlignment = 'center';
            app.Dehazed_RGBSwitchLabel.Enable = 'off';
            app.Dehazed_RGBSwitchLabel.Position = [127 234 86 22];
            app.Dehazed_RGBSwitchLabel.Text = 'Dehazed_RGB';

            % Create HistogramButton
            app.HistogramButton = uibutton(app.LeftPanel, 'push');
            app.HistogramButton.ButtonPushedFcn = createCallbackFcn(app, @HistogramButtonPushed, true);
            app.HistogramButton.Enable = 'off';
            app.HistogramButton.Position = [74 453 100 22];
            app.HistogramButton.Text = 'Histogram';

            % Create Image
            app.Image = uiimage(app.LeftPanel);
            app.Image.Position = [24 7 199 106];
            app.Image.ImageSource = fullfile(pathToMLAPP, '2023-05-26_18h16_08.png');

            % Create RightPanel
            app.RightPanel = uipanel(app.GridLayout);
            app.RightPanel.Layout.Row = 1;
            app.RightPanel.Layout.Column = 2;

            % Create UIAxes_2
            app.UIAxes_2 = uiaxes(app.RightPanel);
            title(app.UIAxes_2, 'Dehazed')
            app.UIAxes_2.XColor = 'none';
            app.UIAxes_2.XTick = [];
            app.UIAxes_2.YColor = 'none';
            app.UIAxes_2.YTick = [];
            app.UIAxes_2.ZColor = 'none';
            app.UIAxes_2.GridColor = 'none';
            app.UIAxes_2.MinorGridColor = 'none';
            app.UIAxes_2.Position = [25 12 350 250];

            % Create UIAxes
            app.UIAxes = uiaxes(app.RightPanel);
            title(app.UIAxes, 'Original')
            app.UIAxes.XColor = 'none';
            app.UIAxes.XTick = [];
            app.UIAxes.YColor = 'none';
            app.UIAxes.YTick = [];
            app.UIAxes.ZColor = 'none';
            app.UIAxes.GridColor = 'none';
            app.UIAxes.MinorGridColor = 'none';
            app.UIAxes.Position = [25 265 350 250];

            % Show the figure after all components are created
            app.UIFigure.Visible = 'on';
        end
    end

    % App creation and deletion
    methods (Access = public)

        % Construct app
        function app = Homomorphic_Image_Dehaze

            % Create UIFigure and components
            createComponents(app)

            % Register the app with App Designer
            registerApp(app, app.UIFigure)

            if nargout == 0
                clear app
            end
        end

        % Code that executes before app deletion
        function delete(app)

            % Delete UIFigure when app is deleted
            delete(app.UIFigure)
        end
    end
end